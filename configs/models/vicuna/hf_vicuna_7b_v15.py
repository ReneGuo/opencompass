from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='vicuna-7b-v1.5-hf',
        path="/mnt/turbocfs/evaluation_pretrain/models/sota/vicuna-7b-v1.5",
        tokenizer_path='/mnt/turbocfs/evaluation_pretrain/models/sota/vicuna-7b-v1.5',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
