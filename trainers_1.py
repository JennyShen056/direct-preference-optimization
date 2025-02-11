######## trainers_1.py #########
import torch

torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from preference_datasets0 import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm
import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple

# from models import MultiHeadPreferenceModel
from models_1 import MultiHeadPreferenceModel


def preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    label_smoothing: float = 0.0,
    ipo: bool = False,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    if ipo:
        losses = (logits - 1 / (2 * beta)) ** 2
    else:
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = (
        beta * (policy_rejected_logps - reference_rejected_logps).detach()
    )

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(
    logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits."""
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100

    labels[labels == -100] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]]
) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor."""
    max_length = max(
        batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1]
    )
    concatenated_batch = {}
    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(
                batch[k], max_length, pad_value=pad_value
            )
    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0,
            )
    return concatenated_batch


class BasicTrainer:
    def __init__(
        self,
        policy: nn.Module,
        config: DictConfig,
        seed: int,
        run_dir: str,
        reference_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """A trainer for a language model, supporting either SFT or DPO training."""
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir

        tokenizer_name_or_path = (
            config.model.tokenizer_name_or_path or config.model.name_or_path
        )
        rank0_print(f"Loading tokenizer {tokenizer_name_or_path}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs)
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == "sft",
        )

        self.policy = policy
        self.reference_model = reference_model

        if isinstance(config.datasets, list):
            # Multi-dimensional case - handled in MultiDimensionalTrainer
            pass
        else:
            # Standard single-dimension case
            self.train_iterator = get_batch_iterator(
                **data_iterator_kwargs,
                split="train",
                n_epochs=config.n_epochs,
                n_examples=config.n_examples,
                batch_size=config.batch_size,
                silent=rank != 0,
                cache_dir=get_local_dir(config.local_dirs),
            )
            self.eval_iterator = get_batch_iterator(
                **data_iterator_kwargs,
                split="test",
                n_examples=config.n_eval_examples,
                batch_size=config.eval_batch_size,
                silent=rank != 0,
                cache_dir=get_local_dir(config.local_dirs),
            )
            self.eval_batches = list(self.eval_iterator)

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""
        ctx = lambda: (
            FSDP.summon_full_params(self.policy, writeback=False, recurse=False)
            if "FSDP" in self.config.trainer
            else contextlib.nullcontext()
        )
        with ctx():
            policy_output = self.policy.generate(
                batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.config.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        if self.config.loss.name in {"dpo", "ipo"}:
            ctx = lambda: (
                FSDP.summon_full_params(
                    self.reference_model, writeback=False, recurse=False
                )
                if "FSDP" in self.config.trainer
                else contextlib.nullcontext()
            )
            with ctx():
                reference_output = self.reference_model.generate(
                    batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=self.config.max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        policy_output = pad_to_length(
            policy_output, self.config.max_length, self.tokenizer.pad_token_id
        )
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(
            policy_output, skip_special_tokens=True
        )

        if self.config.loss.name in {"dpo", "ipo"}:
            reference_output = pad_to_length(
                reference_output, self.config.max_length, self.tokenizer.pad_token_id
            )
            reference_output = all_gather_if_needed(
                reference_output, self.rank, self.world_size
            )
            reference_output_decoded = self.tokenizer.batch_decode(
                reference_output, skip_special_tokens=True
            )
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run forward pass on concatenated chosen and rejected inputs."""
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)
        all_logps = _get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        rejected_logps = all_logps[batch["chosen_input_ids"].shape[0] :]
        return chosen_logps, rejected_logps

    def get_batch_metrics(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        loss_config: DictConfig,
        train=True,
    ):
        """Compute metrics for a single batch."""
        metrics = {}
        train_test = "train" if train else "eval"

        # Add debug info
        for dim_idx, dim_name in enumerate(self.config.datasets):
            chosen_responses = batches[dim_idx]["chosen_response_only"]
            rejected_responses = batches[dim_idx]["rejected_response_only"]

            # Log sample responses
            if self.rank == 0 and train and self.batch_counter % 100 == 0:
                print(f"\nDimension {dim_name} samples:")
                print(f"Chosen: {chosen_responses[0]}")
                print(f"Rejected: {rejected_responses[0]}")

        if loss_config.name in {"dpo", "ipo"}:
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(
                self.policy, batch
            )
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = (
                    self.concatenated_forward(self.reference_model, batch)
                )

            loss_kwargs = {
                "beta": loss_config.beta,
                "reference_free": loss_config.reference_free,
                "label_smoothing": loss_config.label_smoothing,
                "ipo": loss_config.name == "ipo",
            }

            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                **loss_kwargs,
            )

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(
                chosen_rewards, self.rank, self.world_size
            )
            rejected_rewards = all_gather_if_needed(
                rejected_rewards, self.rank, self.world_size
            )
            reward_accuracies = all_gather_if_needed(
                reward_accuracies, self.rank, self.world_size
            )

            metrics[f"rewards_{train_test}/chosen"] = (
                chosen_rewards.cpu().numpy().tolist()
            )
            metrics[f"rewards_{train_test}/rejected"] = (
                rejected_rewards.cpu().numpy().tolist()
            )
            metrics[f"rewards_{train_test}/accuracies"] = (
                reward_accuracies.cpu().numpy().tolist()
            )
            metrics[f"rewards_{train_test}/margins"] = (
                (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            )

            policy_rejected_logps = all_gather_if_needed(
                policy_rejected_logps.detach(), self.rank, self.world_size
            )
            metrics[f"logps_{train_test}/rejected"] = (
                policy_rejected_logps.cpu().numpy().tolist()
            )

        elif loss_config.name == "sft":
            policy_chosen_logits = self.policy(
                batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]
            ).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(
                policy_chosen_logits, batch["chosen_labels"], average_log_prob=False
            )
            losses = -policy_chosen_logps

        policy_chosen_logps = all_gather_if_needed(
            policy_chosen_logps.detach(), self.rank, self.world_size
        )
        metrics[f"logps_{train_test}/chosen"] = (
            policy_chosen_logps.cpu().numpy().tolist()
        )

        all_devices_losses = all_gather_if_needed(
            losses.detach(), self.rank, self.world_size
        )
        metrics[f"loss/{train_test}"] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics

    def train(self):
        """Standard training loop."""
        rank0_print(f"Using {self.config.optimizer} optimizer")
        self.optimizer = getattr(torch.optim, self.config.optimizer)(
            self.policy.parameters(), lr=self.config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                1.0, (step + 1) / (self.config.warmup_steps + 1)
            ),
        )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {"dpo", "ipo"}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        for batch in self.train_iterator:
            if self.example_counter % self.config.eval_every == 0 and (
                self.example_counter > 0 or self.config.do_first_eval
            ):
                self.policy.eval()
                all_eval_metrics = defaultdict(list)

                # Sample if requested
                if self.config.sample_during_eval:
                    all_policy_samples = []
                    all_reference_samples = []
                    policy_text_table = wandb.Table(
                        columns=["step", "prompt", "sample"]
                    )
                    if self.config.loss.name in {"dpo", "ipo"}:
                        reference_text_table = wandb.Table(
                            columns=["step", "prompt", "sample"]
                        )

                # Compute eval metrics
                for eval_batch in (
                    tqdm.tqdm(self.eval_batches, desc="Computing eval metrics")
                    if self.rank == 0
                    else self.eval_batches
                ):
                    local_eval_batch = slice_and_move_batch_for_device(
                        eval_batch, self.rank, self.world_size, self.rank
                    )
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(
                            local_eval_batch, self.config.loss, train=False
                        )

                    for k, v in eval_metrics.items():
                        all_eval_metrics[k].extend(v)

                if self.config.sample_during_eval:
                    if self.config.n_eval_model_samples < self.config.eval_batch_size:
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = (
                            self.config.n_eval_model_samples
                            // self.config.eval_batch_size
                        )
                        sample_batches = self.eval_batches[:n_sample_batches]
                    for eval_batch in (
                        tqdm.tqdm(sample_batches, desc="Generating samples...")
                        if self.rank == 0
                        else sample_batches
                    ):
                        local_eval_batch = slice_and_move_batch_for_device(
                            eval_batch, self.rank, self.world_size, self.rank
                        )
                        policy_samples, reference_samples = self.get_batch_samples(
                            local_eval_batch
                        )

                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)

                        for prompt, sample in zip(eval_batch["prompt"], policy_samples):
                            policy_text_table.add_data(
                                self.example_counter, prompt, sample
                            )
                        if self.config.loss.name in {"dpo", "ipo"}:
                            for prompt, sample in zip(
                                eval_batch["prompt"], reference_samples
                            ):
                                reference_text_table.add_data(
                                    self.example_counter, prompt, sample
                                )

                mean_eval_metrics = {
                    k: sum(v) / len(v) for k, v in all_eval_metrics.items()
                }
                rank0_print(
                    f"eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}"
                )

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                    if self.config.sample_during_eval:
                        wandb.log(
                            {"policy_samples": policy_text_table},
                            step=self.example_counter,
                        )
                        if self.config.loss.name in {"dpo", "ipo"}:
                            wandb.log(
                                {"reference_samples": reference_text_table},
                                step=self.example_counter,
                            )

                if self.example_counter > 0:
                    if not self.config.debug:
                        output_dir = os.path.join(
                            self.run_dir, f"step-{self.example_counter}"
                        )
                        rank0_print(f"creating checkpoint to write to {output_dir}...")
                        self.save(output_dir, mean_eval_metrics)

            self.policy.train()
            start_time = time.time()
            batch_metrics = defaultdict(list)

            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(
                    batch,
                    microbatch_idx,
                    self.config.gradient_accumulation_steps,
                    self.rank,
                )
                local_microbatch = slice_and_move_batch_for_device(
                    global_microbatch, self.rank, self.world_size, self.rank
                )
                loss, metrics = self.get_batch_metrics(
                    local_microbatch, self.config.loss, train=True
                )
                (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics["examples_per_second"].append(examples_per_second)
            batch_metrics["grad_norm"].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if (
                last_log is None
                or time.time() - last_log > self.config.minimum_log_interval_secs
            ):
                mean_train_metrics = {
                    k: sum(v) / len(v) for k, v in batch_metrics.items()
                }
                mean_train_metrics["counters/examples"] = self.example_counter
                mean_train_metrics["counters/updates"] = self.batch_counter
                rank0_print(
                    f"train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}"
                )

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        ).item()

    def write_state_dict(
        self,
        step: int,
        state: Dict[str, torch.Tensor],
        metrics: Dict,
        filename: str,
        dir_name: Optional[str] = None,
    ):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f"LATEST")

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f"writing checkpoint to {output_path}...")
        torch.save(
            {
                "step_idx": step,
                "state": state,
                "metrics": metrics if metrics is not None else {},
            },
            output_path,
        )

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""
        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(
            self.example_counter, policy_state_dict, metrics, "policy.pt", output_dir
        )
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(
            self.example_counter,
            optimizer_state_dict,
            metrics,
            "optimizer.pt",
            output_dir,
        )
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(
            self.example_counter,
            scheduler_state_dict,
            metrics,
            "scheduler.pt",
            output_dir,
        )


class FSDPTrainer(BasicTrainer):
    def __init__(
        self,
        policy: nn.Module,
        config: DictConfig,
        seed: int,
        run_dir: str,
        reference_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs."""
        super().__init__(
            policy, config, seed, run_dir, reference_model, rank, world_size
        )
        assert (
            config.model.block_name is not None
        ), "must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP"

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={wrap_class},
        )

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False,
        )

        rank0_print("Sharding policy...")
        mp_dtype = (
            getattr(torch, config.model.fsdp_policy_mp)
            if config.model.fsdp_policy_mp is not None
            else None
        )
        policy_mp_policy = MixedPrecision(
            param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype
        )
        self.policy = FSDP(
            policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy
        )

        if config.activation_checkpointing:
            rank0_print("Attempting to enable activation checkpointing...")
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )

                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print("FSDP activation checkpointing not available:", e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print("Applying activation checkpointing wrapper to policy...")
                apply_activation_checkpointing(
                    self.policy,
                    checkpoint_wrapper_fn=non_reentrant_wrapper,
                    check_fn=check_fn,
                )
                rank0_print("FSDP activation checkpointing enabled!")

        if config.loss.name in {"dpo", "ipo"}:
            rank0_print("Sharding reference model...")
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)

        print("Loaded model on rank", rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()

    def save(self, output_dir=None, metrics=None):
        """Save FSDP policy, optimizer, and scheduler state."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy
        ):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(
                self.example_counter,
                policy_state_dict,
                metrics,
                "policy.pt",
                output_dir,
            )
        del policy_state_dict
        dist.barrier()

        save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            self.policy,
            StateDictType.FULL_STATE_DICT,
            optim_state_dict_config=save_policy,
        ):
            optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

        if self.rank == 0:
            self.write_state_dict(
                self.example_counter,
                optimizer_state_dict,
                metrics,
                "optimizer.pt",
                output_dir,
            )
        del optimizer_state_dict
        dist.barrier()

        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(
                self.example_counter,
                scheduler_state_dict,
                metrics,
                "scheduler.pt",
                output_dir,
            )
        dist.barrier()


class MultiDimensionalFSDPTrainer(FSDPTrainer):
    def __init__(
        self,
        policy: nn.Module,
        config: DictConfig,
        seed: int,
        run_dir: str,
        reference_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """A trainer that handles multiple preference dimensions with vectorized heads."""
        super().__init__(
            policy, config, seed, run_dir, reference_model, rank, world_size
        )

        # Number of preference dimensions (heads)
        self.n_dims = len(config.datasets)

        # Create separate iterators for each dimension
        self.train_iterators = [
            get_batch_iterator(
                names=[dataset],
                tokenizer=self.tokenizer,
                shuffle=True,
                max_length=config.max_length,
                max_prompt_length=config.max_prompt_length,
                sft_mode=False,
                split="train",
                n_epochs=config.n_epochs,
                n_examples=config.n_examples,
                batch_size=config.batch_size,
                silent=rank != 0,
                cache_dir=get_local_dir(config.local_dirs),
            )
            for dataset in config.datasets
        ]

        # Create eval iterators for each dimension
        self.eval_iterators = [
            get_batch_iterator(
                names=[dataset],
                tokenizer=self.tokenizer,
                shuffle=False,
                max_length=config.max_length,
                max_prompt_length=config.max_prompt_length,
                sft_mode=False,
                split="test",
                n_examples=config.n_eval_examples,
                batch_size=config.eval_batch_size,
                silent=rank != 0,
                cache_dir=get_local_dir(config.local_dirs),
            )
            for dataset in config.datasets
        ]
        self.eval_batches = [list(iterator) for iterator in self.eval_iterators]

    def concatenated_forward_multi_dim(
        self,
        model: nn.Module,
        batches: List[Dict[str, Union[List, torch.LongTensor]]],
        train: bool = True,
    ) -> Tuple[List[torch.FloatTensor], List[torch.FloatTensor]]:
        """Run forward pass for all dimensions in parallel by stacking inputs."""
        # Find maximum sequence length across all batches
        max_length = max(
            max(
                batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1]
            )
            for batch in batches
        )

        # Initialize lists to store inputs for each dimension
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        batch_sizes = []

        # Process each dimension's batch
        for batch in batches:
            # Get concatenated inputs for this dimension
            concatenated_batch = concatenated_inputs(batch)

            # Store batch size for this dimension
            batch_sizes.append(concatenated_batch["concatenated_input_ids"].shape[0])

            # Pad to max_length
            padded_input_ids = pad_to_length(
                concatenated_batch["concatenated_input_ids"],
                max_length,
                pad_value=self.tokenizer.pad_token_id,
            )
            padded_attention_mask = pad_to_length(
                concatenated_batch["concatenated_attention_mask"],
                max_length,
                pad_value=0,
            )
            padded_labels = pad_to_length(
                concatenated_batch["concatenated_labels"], max_length, pad_value=-100
            )

            all_input_ids.append(padded_input_ids)
            all_attention_masks.append(padded_attention_mask)
            all_labels.append(padded_labels)

        # Stack all dimensions together
        stacked_input_ids = torch.cat(all_input_ids, dim=0)
        stacked_attention_mask = torch.cat(all_attention_masks, dim=0)
        stacked_labels = torch.cat(all_labels, dim=0)

        # Forward pass through model
        outputs = model(stacked_input_ids, attention_mask=stacked_attention_mask)
        all_logits = outputs.logits.to(torch.float32)

        # Split results by dimension
        chosen_logps = []
        rejected_logps = []
        current_idx = 0

        for batch_size in batch_sizes:
            # Get this dimension's logits
            dim_logits = all_logits[current_idx : current_idx + batch_size]
            dim_labels = stacked_labels[current_idx : current_idx + batch_size]

            # Calculate log probabilities
            dim_logps = _get_batch_logps(dim_logits, dim_labels, average_log_prob=False)

            # Split into chosen and rejected
            chosen_size = batch_size // 2
            chosen_logps.append(dim_logps[:chosen_size])
            rejected_logps.append(dim_logps[chosen_size:])

            current_idx += batch_size

        return chosen_logps, rejected_logps

    def get_batch_metrics(
        self,
        batches: List[Dict[str, Union[List, torch.LongTensor]]],
        loss_config: DictConfig,
        train=True,
    ) -> Tuple[torch.FloatTensor, Dict]:
        """Compute metrics for all preference dimensions."""
        metrics = {}
        train_test = "train" if train else "eval"

        # Get log probabilities for all dimensions
        policy_chosen_logps, policy_rejected_logps = (
            self.concatenated_forward_multi_dim(self.policy, batches, train=train)
        )

        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps = (
                self.concatenated_forward_multi_dim(
                    self.reference_model, batches, train=train
                )
            )

        total_loss = 0
        for dim_idx in range(self.n_dims):
            dim_name = self.config.datasets[dim_idx]

            # Compute DPO loss for each dimension
            dim_losses, dim_chosen_rewards, dim_rejected_rewards = preference_loss(
                policy_chosen_logps[dim_idx],
                policy_rejected_logps[dim_idx],
                reference_chosen_logps[dim_idx],
                reference_rejected_logps[dim_idx],
                beta=loss_config.beta,
                reference_free=loss_config.reference_free,
                label_smoothing=loss_config.label_smoothing,
                ipo=loss_config.name == "ipo",
            )

            # Add dimension's contribution to total loss
            total_loss += dim_losses.mean() / self.n_dims

            # Record metrics for this dimension
            reward_accuracies = (dim_chosen_rewards > dim_rejected_rewards).float()
            reward_margins = dim_chosen_rewards - dim_rejected_rewards

            metrics[f"{dim_name}/loss/{train_test}"] = (
                all_gather_if_needed(dim_losses.detach(), self.rank, self.world_size)
                .cpu()
                .numpy()
                .tolist()
            )
            metrics[f"{dim_name}/rewards_{train_test}/chosen"] = (
                all_gather_if_needed(dim_chosen_rewards, self.rank, self.world_size)
                .cpu()
                .numpy()
                .tolist()
            )
            metrics[f"{dim_name}/rewards_{train_test}/rejected"] = (
                all_gather_if_needed(dim_rejected_rewards, self.rank, self.world_size)
                .cpu()
                .numpy()
                .tolist()
            )
            metrics[f"{dim_name}/rewards_{train_test}/accuracies"] = (
                all_gather_if_needed(reward_accuracies, self.rank, self.world_size)
                .cpu()
                .numpy()
                .tolist()
            )
            metrics[f"{dim_name}/rewards_{train_test}/margins"] = (
                all_gather_if_needed(reward_margins, self.rank, self.world_size)
                .cpu()
                .numpy()
                .tolist()
            )
            metrics[f"{dim_name}/logps_{train_test}/chosen"] = (
                all_gather_if_needed(
                    policy_chosen_logps[dim_idx].detach(), self.rank, self.world_size
                )
                .cpu()
                .numpy()
                .tolist()
            )
            metrics[f"{dim_name}/logps_{train_test}/rejected"] = (
                all_gather_if_needed(
                    policy_rejected_logps[dim_idx].detach(), self.rank, self.world_size
                )
                .cpu()
                .numpy()
                .tolist()
            )

        return total_loss, metrics

    def _get_unwrapped_model(self, model):
        """Recursively unwrap FSDP model to get the actual model."""
        if hasattr(model, "_fsdp_wrapped_module"):
            return self._get_unwrapped_model(model._fsdp_wrapped_module)
        return model

    def get_per_head_grads(self):
        """Get gradient norms for each preference head."""
        # Get the actual model
        unwrapped_model = self._get_unwrapped_model(self.policy)

        head_grads = []
        # Access preference heads through unwrapped model
        if hasattr(unwrapped_model, "preference_heads"):
            for head_idx, head in enumerate(unwrapped_model.preference_heads):
                total_norm = 0.0
                if hasattr(head, "weight") and head.weight.grad is not None:
                    total_norm += head.weight.grad.norm().item() ** 2
                if (
                    hasattr(head, "bias")
                    and head.bias is not None
                    and head.bias.grad is not None
                ):
                    total_norm += head.bias.grad.norm().item() ** 2
                head_grads.append(total_norm**0.5)
        return head_grads

    def train(self):
        """Training loop that handles multiple preference dimensions."""
        rank0_print(f"Using {self.config.optimizer} optimizer")
        self.optimizer = getattr(torch.optim, self.config.optimizer)(
            self.policy.parameters(), lr=self.config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                1.0, (step + 1) / (self.config.warmup_steps + 1)
            ),
        )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {"dpo", "ipo"}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        # Get iterators for all dimensions
        iterators = zip(*self.train_iterators)

        for batches in iterators:
            # Training step
            self.policy.train()
            start_time = time.time()
            batch_metrics = defaultdict(list)

            # Process microbatches with gradient accumulation
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                # Process microbatch for all dimensions
                global_microbatches = [
                    slice_and_move_batch_for_device(
                        batch,
                        microbatch_idx,
                        self.config.gradient_accumulation_steps,
                        self.rank,
                    )
                    for batch in batches
                ]
                local_microbatches = [
                    slice_and_move_batch_for_device(
                        batch, self.rank, self.world_size, self.rank
                    )
                    for batch in global_microbatches
                ]

                with torch.cuda.amp.autocast(enabled=True):
                    loss, metrics = self.get_batch_metrics(
                        local_microbatches, self.config.loss, train=True
                    )
                    scaled_loss = loss / self.config.gradient_accumulation_steps

                scaled_loss.backward()

                # Get per-head gradients after backward pass
                with FSDP.summon_full_params(
                    self.policy, writeback=False, recurse=True
                ):
                    head_grads = self.get_per_head_grads()

                # Log per-dimension gradients
                for dim_idx, (dim_name, grad_norm) in enumerate(
                    zip(self.config.datasets, head_grads)
                ):
                    metrics[f"{dim_name}/head_grad_norm"] = [grad_norm]

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics["examples_per_second"].append(examples_per_second)
            batch_metrics["grad_norm"].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if (
                last_log is None
                or time.time() - last_log > self.config.minimum_log_interval_secs
            ):
                mean_train_metrics = {
                    k: sum(v) / len(v) for k, v in batch_metrics.items()
                }
                mean_train_metrics["counters/examples"] = self.example_counter
                mean_train_metrics["counters/updates"] = self.batch_counter
                rank0_print(
                    f"train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}"
                )

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
