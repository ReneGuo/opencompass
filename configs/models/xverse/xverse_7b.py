from opencompass.models import HuggingFaceCausalLM

xverse_7b = dict(
    type=HuggingFaceCausalLM,
    abbr='xverse-7b',
    path="/mnt/turbocfs/evaluation_pretrain/models/llama7b_v11/llama7b-v11-2667b",
    tokenizer_path='/mnt/turbocfs/evaluation_pretrain/models/llama7b_v11/llama7b-v11-2667b',
    tokenizer_kwargs=dict(),
    max_out_len=100,
    max_seq_len=8192,
    batch_size=8,
    model_kwargs=dict(trust_remote_code=True, device_map='auto'),
    batch_padding=False,  # if false, inference with for-loop without batch padding
    run_cfg=dict(num_gpus=1, num_procs=1),
)
