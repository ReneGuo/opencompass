from mmengine.config import read_base

with read_base():
    from .datasets.winogrande.winogrande_ppl import winogrande_datasets
    from .datasets.piqa.piqa_ppl import piqa_datasets
    from .datasets.hellaswag.hellaswag_ppl import hellaswag_datasets
    from .models.xverse.xverse_7b import xverse_7b
    from .models.xverse.xverse_7b_chat import xverse_7b_chat
    from .models.falcon.hf_falcon_7b import models as falcon_models
    from .models.vicuna.hf_vicuna_7b_v15 import models as vicuna_models

# datasets = [*winogrande_datasets, *piqa_datasets, *hellaswag_datasets]
datasets = [*winogrande_datasets]

models = vicuna_models
