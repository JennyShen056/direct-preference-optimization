# multi_head_model.py
import torch
import torch.nn as nn
from transformers import PreTrainedModel


class MultiHeadCausalLM(nn.Module):
    def __init__(self, base_model: PreTrainedModel, num_heads: int):
        super().__init__()
        self.base_model = base_model
        self.num_heads = num_heads

        # Retrieve dimensions from the base model configuration
        hidden_size = base_model.config.hidden_size
        vocab_size = base_model.config.vocab_size

        # Get the target device
        target_device = next(base_model.parameters()).device

        # Create and move heads to the same device as base model
        self.heads = nn.ModuleList()
        if hasattr(base_model, "lm_head"):
            self.heads.append(base_model.lm_head.to(target_device))
        else:
            self.heads.append(
                nn.Linear(hidden_size, vocab_size, bias=False).to(target_device)
            )

        for _ in range(1, num_heads):
            new_head = nn.Linear(hidden_size, vocab_size, bias=False).to(target_device)
            new_head.weight.data.copy_(self.heads[0].weight.data)
            self.heads.append(new_head)

        if hasattr(base_model, "lm_head"):
            del self.base_model.lm_head

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Ensure inputs are on the same device as the model
        device = next(self.base_model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Forward pass through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        hidden_states = outputs.hidden_states[-1]

        # Apply heads and ensure outputs are on the same device
        logits_per_head = []
        for head in self.heads:
            logits = head(hidden_states)  # This should maintain device placement
            logits_per_head.append(logits)

        multi_head_logits = torch.stack(logits_per_head, dim=0)
        return {"logits": multi_head_logits}

    def generate(self, *args, **kwargs):
        # Ensure inputs are on the correct device
        device = next(self.base_model.parameters()).device
        if "input_ids" in kwargs:
            kwargs["input_ids"] = kwargs["input_ids"].to(device)
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = kwargs["attention_mask"].to(device)

        # Save any original lm_head
        original_lm_head = getattr(self.base_model, "lm_head", None)
        # Inject head 0
        self.base_model.lm_head = self.heads[0]
        generated = self.base_model.generate(*args, **kwargs)
        # Restore original head or delete injected one
        if original_lm_head is not None:
            self.base_model.lm_head = original_lm_head
        else:
            del self.base_model.lm_head
        return generated

    @property
    def device(self):
        """Get the device of the model."""
        return next(self.base_model.parameters()).device

    def to(self, device):
        """Move the model to the specified device."""
        self.base_model = self.base_model.to(device)
        self.heads = nn.ModuleList([head.to(device) for head in self.heads])
        return self
