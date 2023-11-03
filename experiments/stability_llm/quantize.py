import os
import random
import shutil
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
from optimum.exporters import TasksManager
from optimum.intel.openvino import OVConfig
from optimum.intel.openvino import OVModelForCausalLM
from optimum.intel.openvino.quantization import OVQuantizer
from optimum.utils import NormalizedConfigManager
from optimum.utils import NormalizedTextConfig
from transformers import AutoConfig
from transformers import AutoTokenizer

from datasets import load_dataset
from nncf import Dataset
from nncf import IgnoredScope
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters

random.seed(10)


def transform_func(item, tokenizer, max_input_length):
    text = item["text"]
    tokens = tokenizer(text, max_length=max_input_length, truncation=True)
    # return tokens['input_ids'], tokens['attention_mask']

    res = {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}

    # res.update(gen_pkv(12, 64)) #opt-125
    # res.update(gen_pkv(32, 128))
    # res.update(gen_pkv(32, 80, 32))
    # res.update(gen_pkv(32, 128, 32)) #llama

    return res


def quantize(model_id, save_path, dataset_path, dataset_name, alpha=0.95):
    print(f"Quantizing {model_id}")

    TasksManager._SUPPORTED_MODEL_TYPE["stablelm-epoch"] = TasksManager._SUPPORTED_MODEL_TYPE["llama"]
    NormalizedConfigManager._conf["stablelm_epoch"] = NormalizedTextConfig.with_args(
        num_layers="num_hidden_layers", num_attention_heads="num_attention_heads"
    )

    ov_model = OVModelForCausalLM.from_pretrained(
        # "/home/susloval/work/projects/nncf/stabilityai_stablelm-3b-4e1t_fp32/channel_aligment",
        save_path,
        config=AutoConfig.from_pretrained(model_id, trust_remote_code=True),
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = 512

    quantizer = OVQuantizer.from_pretrained(ov_model)

    # calibration_dataset = load_dataset(dataset_path, dataset_name, split='train')
    # nncf_dataset = Dataset(calibration_dataset, partial(transform_func, tokenizer=tokenizer))

    calibration_dataset = quantizer.get_calibration_dataset(
        dataset_path,
        dataset_config_name=dataset_name,
        preprocess_function=partial(transform_func, tokenizer=tokenizer, max_input_length=128),
        num_samples=300,
        dataset_split="train",
    )

    # calibration_dataset = quantizer.get_calibration_dataset(
    #     dataset_path,
    #     #dataset_config_name=dataset_name,
    #     preprocess_function=partial(transform_func, tokenizer=tokenizer, max_input_length=256),
    #     num_samples=300,
    #     dataset_split="validation",
    # )

    config = OVConfig()

    ignored_scope = IgnoredScope(
        patterns=[
            "__module.model.model.layers.*.self_attn/aten::matmul/MatMul_",
            # "__module.model.model.layers.31.mlp.down_proj/aten::linear/MatMul_.*",
            # "__module.model.model.layers.6.mlp.gate_proj/aten::linear/MatMul_.*",
            # "__module.model.model.layers.6.mlp.up_proj/aten::linear/MatMul_.*",
            # "__module.model.model.layers.6.mlp.down_proj/aten::linear/MatMul_.*",
            # "__module.model.model.layers.29.mlp.down_proj/aten::linear/MatMul_.*",
            # "__module.model.model.layers.30.mlp.down_proj/aten::linear/MatMul_.*",
            # "__module.model.model.layers.3.mlp.down_proj/aten::linear/MatMul_.*",
            # ".*down_proj.*",
        ]
    )

    quantizer.quantize(
        calibration_dataset,
        save_path + f"/all_w8a8_{dataset_path}_{alpha}",
        quantization_config=config,
        ignored_scope=ignored_scope,
        advanced_parameters=AdvancedQuantizationParameters(
            disable_channel_alignment=False, disable_bias_correction=True, smooth_quant_alpha=alpha
        ),
    )


# alphas = [0.7, 0.5, 0.25, 0.15, 0.95]
alphas = [0.55, 0.6, 0.65, 0.7, 0.9, 0.95]

# alphas = [0.7]
for alpha in alphas:
    quantize(
        "stabilityai/stablelm-3b-4e1t",
        "/home/susloval/work/projects/nncf/stabilityai_stablelm-3b-4e1t_fp32",
        "wikitext",
        "wikitext-2-v1",
        alpha,
    )
# quantize("stabilityai/stablelm-3b-4e1t", "/home/aanuf/proj/ov_compression/stabilityai_stablelm-3b-4e1t_fp32", 'squad', '')
