import time
from functools import partial

from optimum.intel.openvino import OVModelForCausalLM
from optimum.intel.openvino import OVQuantizer
from transformers import AutoTokenizer

import nncf

MODEL_ID = "databricks/dolly-v2-3b"
# MODEL_ID = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
# MODEL_ID = "EleutherAI/pythia-410m-deduped"
# MODEL_ID = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True, use_cache=True)

model.save_pretrained("dolly_openvino")
model = OVModelForCausalLM.from_pretrained("dolly_openvino", use_cache=True)


def preprocess_fn(examples, tokenizer):
    data = tokenizer(examples["sentence"])
    return data


quantizer = OVQuantizer.from_pretrained(model)
dummy_dataset = quantizer.get_calibration_dataset(
    "glue",
    dataset_config_name="sst2",
    preprocess_function=partial(preprocess_fn, tokenizer=tokenizer),
    num_samples=10,
    dataset_split="train",
    preprocess_batch=True,
)

tic = time.perf_counter()
quantizer.quantize(
    calibration_dataset=dummy_dataset,
    save_directory="dolly_full_quantized",
    advanced_parameters=nncf.AdvancedQuantizationParameters(backend_params={"use_pot": True}),
)
toc = time.perf_counter()

print(f"Quantization wall time: {toc - tic:0.4f} seconds")
