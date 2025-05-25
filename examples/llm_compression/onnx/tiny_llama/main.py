# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from pathlib import Path

import onnx

# from optimum.intel.openvino import OVModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

import nncf

ROOT = Path(__file__).parent.resolve()


def compress_model(onnx_model_path: Path):
    onnx_model = onnx.load(onnx_model_path, load_external_data=False)

    # Comment this text to turn off model optimization and measure performance of baseline model
    compressed_onnx_model = nncf.compress_weights(
        onnx_model,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=0.8,
        advanced_parameters=nncf.AdvancedCompressionParameters(
            backend_params={"external_data_dir": onnx_model_path.parent}
        ),
    )

    # replace original model on the compressed model
    onnx.save(compressed_onnx_model, onnx_model_path, save_as_external_data=True)


def main():
    MODEL_ID = "PY007/TinyLlama-1.1B-Chat-v0.3"
    OUTPUT_DIR = ROOT / "tinyllama_compressed"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = ORTModelForCausalLM.from_pretrained(MODEL_ID, export=True)

    # save pretrained model to the output directory
    model.save_pretrained(OUTPUT_DIR)

    # compress model
    compress_model(OUTPUT_DIR / "model.onnx")

    # infer model
    # model = OVModelForCausalLM.from_pretrained(OUTPUT_DIR, export=False, from_onnx=True)
    # model = ORTModelForCausalLM.from_pretrained(OUTPUT_DIR, provider="OpenVINOExecutionProvider")
    model = ORTModelForCausalLM.from_pretrained(OUTPUT_DIR)
    input_ids = tokenizer("What is PyTorch?", return_tensors="pt").to(device=model.device)

    start_t = time.time()
    output = model.generate(**input_ids, max_new_tokens=100)
    print("Elapsed time: ", time.time() - start_t)

    output_text = tokenizer.decode(output[0])
    print(output_text)
    return output_text


if __name__ == "__main__":
    main()
