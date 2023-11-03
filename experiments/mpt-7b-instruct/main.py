import cProfile
import time
from functools import partial

from optimum.intel.openvino import OVModelForCausalLM
from optimum.intel.openvino import OVQuantizer
from transformers import AutoTokenizer
import perfcounter

from nncf.common.factory import NNCFGraphFactory
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters

MODEL_ID = "mosaicml/mpt-7b-instruct"
model_path = "mpt-7b-instruct"


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True, use_cache=True)

model.save_pretrained(model_path)
model = OVModelForCausalLM.from_pretrained(model_path, use_cache=True)


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


# with cProfile.Profile() as pr:
#     perfcounter.perf_start("graph")
#     nncf_graph = NNCFGraphFactory.create(quantizer.model.model)
#     perfcounter.perf_end("graph")
#     pr.dump_stats("temp.dat")

# with cProfile.Profile() as pr:
#    tic = time.perf_counter()
perfcounter.perf_start("all")
quantizer.quantize(
    calibration_dataset=dummy_dataset,
    save_directory=model_path + "_quantized",
    # advanced_parameters=AdvancedQuantizationParameters(backend_params={"compress_weights": False}),
)
perfcounter.perf_end("all")
print(perfcounter.perf_report())
# toc = time.perf_counter()
# print(f"Quantization wall time: {toc - tic:0.4f} seconds")
# pr.dump_stats("temp.dat")
