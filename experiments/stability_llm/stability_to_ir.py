from optimum.exporters import TasksManager
from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


TasksManager._SUPPORTED_MODEL_TYPE["stablelm-epoch"] = TasksManager._SUPPORTED_MODEL_TYPE["llama"]
NormalizedConfigManager._conf["stablelm_epoch"] = NormalizedTextConfig.with_args(
    num_layers="num_hidden_layers", num_attention_heads="num_attention_heads"
)

test = False
save_path = "stabilityai_stablelm-3b-4e1t_fp32"
model_path = "stabilityai/stablelm-3b-4e1t"  # "stablellm_share_pt"
tokenizer = AutoTokenizer.from_pretrained(model_path)

if test:
    ov_model = OVModelForCausalLM.from_pretrained(
        save_path, config=AutoConfig.from_pretrained(model_path, trust_remote_code=True), trust_remote_code=True
    )

    inputs = tokenizer("The weather is always wonderful", return_tensors="pt").to("cpu")
    tokens = ov_model.generate(
        **inputs,
        # max_new_tokens=64,
        # temperature=0.75,
        # top_p=0.95,
        # do_sample=True,
    )
    # # print(tokenizer.decode(tokens[0], skip_special_tokens=True))
    # hf_model = AutoModelForCausalLM.from_pretrained(model_path, config=AutoConfig.from_pretrained(model_path, trust_remote_code=True), trust_remote_code=True)
    # inputs = tokenizer("What is openvino ?", return_tensors="pt").to("cpu")
    # tokens = hf_model.generate(
    # **inputs,
    # #max_new_tokens=64,
    # #temperature=0.75,
    # #top_p=0.95,
    # #do_sample=True,
    # )
    # print(tokenizer.decode(tokens[0], skip_special_tokens=True))
else:
    ov_model = OVModelForCausalLM.from_pretrained(
        model_path,
        config=AutoConfig.from_pretrained(model_path, trust_remote_code=True),
        trust_remote_code=True,
        export=True,
    )
    ov_model.save_pretrained(save_path)

    inputs = tokenizer("What is openvino ?", return_tensors="pt").to("cpu")
    tokens = ov_model.generate(
        **inputs,
        # max_new_tokens=64,
        # temperature=0.75,
        # top_p=0.95,
        # do_sample=True,
    )
    print(tokenizer.decode(tokens[0], skip_special_tokens=True))
