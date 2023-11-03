# import torch
# from transformers import AutoModel
# from transformers import AutoTokenizer

# import nncf

# # THUDM/chatglm2-6b
# # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
# # model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda()

# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()

# from torch.fx import symbolic_trace

# symbolic_traced: torch.fx.GraphModule = symbolic_trace(model)

# compressed_model = nncf.compress_weights(model)

# # tricky check that all weights compressed
# for name, parameter in compressed_model.named_parameters():
#     if parameter.dtype not in [torch.uint8, torch.int8] and "bias" not in name and "layernorm" not in name:
#         print(name)

# response, history = compressed_model.chat(tokenizer, "Hi!", history=[])

# print(response)

# response, history = compressed_model.chat(tokenizer, "How many keys are there on a piano?", history=history)

# print(response)


import nncf
import torch
import openvino as ov

from torch.utils.tensorboard import SummaryWriter


def dump_torch_graph(model, input_to_model, logdir):
    writer = SummaryWriter(logdir)
    writer.add_graph(model, input_to_model=input_to_model)
    writer.close()


from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, torchscript=True).float().cpu()

# a = torch.compile(model)

# compressed_model = nncf.compress_weights(model)
compressed_model = model

# tricky check that all weights compressed
for name, parameter in compressed_model.named_parameters():
    if parameter.dtype not in [torch.uint8, torch.int8] and "bias" not in name and "layernorm" not in name:
        print(name)

call_args_cache = {}


def make_call_wrapper(args_cache, model_call):
    def call_fn(_, *args, **kwargs):
        if not args_cache:
            args_cache["args"] = args
            args_cache["kwargs"] = kwargs
        return model_call(*args, **kwargs)

    return call_fn


model_call = getattr(model, "__call__")
setattr(model.__class__, "__call__", make_call_wrapper(call_args_cache, model_call))
response, history = model.chat(tokenizer, "Hi!", history=[])
setattr(model.__class__, "__call__", model_call)

tensor_kwargs = {}
obj_kwargs = {}

if "return_dict" in call_args_cache["kwargs"]:
    call_args_cache["kwargs"]["return_dict"] = False
    call_args_cache["kwargs"]["inputs_embeds"] = None
    call_args_cache["kwargs"]["use_cache"] = None

from typing import List


def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward


# Reset since we are using a different backend.
torch._dynamo.reset()

opt_model = torch.compile(model, backend=custom_backend)
opt_model(**call_args_cache["kwargs"])


for key, value in call_args_cache["kwargs"].items():
    if isinstance(value, torch.Tensor):
        tensor_kwargs[key] = value
    else:
        obj_kwargs[key] = value


def make_forward_call(args_cache, forward_call):
    def _forward_fn(**kwargs):
        return forward_call(**kwargs)

    _locals = {}
    _globals = {}
    _globals["_forward_fn"] = _forward_fn

    args = []
    call_args = []
    for key, value in args_cache.items():
        if isinstance(value, torch.Tensor):
            args.append(f"{key}=None")
            call_args.append(f"{key}={key}")
        else:
            global_key = f"global_{key}"
            _globals[global_key] = value
            call_args.append(f"{key}={global_key}")

    str_args = ", ".join(args)
    str_call_args = ", ".join(call_args)

    exec(
        f"def forward_fn(_, {str_args}): return _forward_fn({str_call_args})",
        _globals,
        _locals,
    )

    return _locals["forward_fn"]


forward_call = getattr(model, "forward")
setattr(model.__class__, "forward", make_forward_call(call_args_cache["kwargs"], forward_call))
ov_model = ov.convert_model(model, example_input=tensor_kwargs)
ov.save_model(ov_model, "chatglm/chatglm.xml", compress_to_fp16=False)
# traced_model = torch.jit.trace(model, example_kwarg_inputs=tensor_kwargs)
setattr(model.__class__, "forward", forward_call)


response, history = model.chat(tokenizer, "Hi!", history=[])

print(response)

response, history = model.chat(tokenizer, "How many keys are there on a piano?", history=history)

print(response)
