import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import nncf
from nncf.experimental.torch.quantization.quantize_model import create_nncf_network

model = torchvision.models.swin_t()
# model #nncf.compress_weights(model)


# ov_model = convert_model(model, example_input=torch.rand(1,3,224,224))

example_inpput = torch.rand(1, 3, 224, 224)
result = model(example_inpput)
compressed_model = nncf.compress_weights(model)


def dump_torch_graph(model, input_to_model, logdir):
    writer = SummaryWriter(logdir)
    writer.add_graph(model, input_to_model=input_to_model)
    writer.close()


dump_torch_graph(compressed_model, example_inpput, "runs/swin/compressed")

wrapped_compressed_model = create_nncf_network(compressed_model, example_inpput)
wrapped_compressed_model.nncf.get_graph().visualize_graph("runs/swin/compressed_graph.dot")
