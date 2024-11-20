# python -u train.py \
#     model=pythia69 \
#     datasets=[hh_helpful,hh_harmless] \
#     dimensions=["helpful","harmless"] \
#     loss=dpo loss.beta=0.1 \
#     model.archive=/path/to/checkpoint/from/sft/step-XXXX/policy.pt \
#     exp_name=multi_dimensional_training \
#     gradient_accumulation_steps=2 batch_size=32 eval_batch_size=32 \
#     trainer=FSDPTrainer sample_during_eval=false


# python -u train.py model=pythia69 datasets=[hh-helpful,hh-harmless] loss=dpo loss.beta=0.1 model.archive=/path/to/checkpoint/from/sft/step-XXXX/policy.pt exp_name=anthropic_dpo_pythia69 gradient_accumulation_steps=2 batch_size=32 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false dimensions=[helpful,harmless]
