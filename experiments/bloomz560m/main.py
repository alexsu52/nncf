import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nncf.experimental.torch.quantization.quantize_model import create_nncf_network

import nncf
import openvino as ov


MODEL_ID = "bigscience/bloomz-560m"
model_path = "bloomz-560m"

from torch.utils.tensorboard import SummaryWriter


def dump_torch_graph(model, input_to_model, logdir):
    writer = SummaryWriter(logdir)
    writer.add_graph(model, input_to_model=input_to_model)
    writer.close()


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")  # , torchscript=True)

# compressed_model = nncf.compress_weights(model)

inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt").to("cuda")

# dump_torch_graph(model, inputs, "runs/bloom/")
# script = torch.jit.trace(model, inputs)
# ov_model = ov.convert_model(model, example_input=inputs)

wrapped_model = create_nncf_network(model, inputs)
wrapped_model.nncf.get_graph().visualize_graph("runs/bloom/original_graph.dot")


outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
