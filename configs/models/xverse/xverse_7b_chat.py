from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='Human: ', end='<|endoftext|>\n\n'),
        dict(role="BOT", begin="Assistant: ", generate=True),
    ],
)

xverse_7b_chat = dict(
    type=HuggingFaceCausalLM,
    abbr='xverse-7b-chat',
    path="/mnt/turbocfs/evaluation_pretrain/models/xverse-7b-2667b-8k-0925",
    tokenizer_path='/mnt/turbocfs/evaluation_pretrain/models/xverse-7b-2667b-8k-0925',
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True,
    ),
    max_out_len=100,
    max_seq_len=8192,
    batch_size=8,
    meta_template=_meta_template,
    model_kwargs=dict(
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True,
    ),
    run_cfg=dict(num_gpus=1, num_procs=1),
)
