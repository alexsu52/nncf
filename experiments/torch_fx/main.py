import os
from typing import Dict

import openvino.torch
import perfcounter
import torch
import torch.fx
from optimum.modeling_base import OptimizedModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
from transformers import GenerationMixin
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.cache_utils import StaticCacheConfig
from transformers.integrations.executorch import TorchExportableModuleWithStaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast

import datasets
import nncf
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching


class FXAutoModelForCausalLM(OptimizedModel, GenerationMixin):
    def __init__(
        self,
        model: torch.fx.GraphModule,
        config: PretrainedConfig,
        device: str = "cpu",
        compile: bool = True,
        backend: str = None,
        dtype=torch.float32,
    ):
        super().__init__(model, config)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.main_input_name = "input_ids"
        self.compile = compile
        self.backend = backend
        self._device = device.upper()
        self._dtype = dtype
        self._cached_prefill_input_ids = None
        self._cached_cache_position = None

        if self.compile:
            if backend is None or backend != "openvino":
                prefil_args = {"fullgraph": True, "dynamic": True}
                decode_one_token_args = {"fullgraph": True, "mode": "reduce-overhead"}
                if backend is not None:
                    prefil_args["backend"] = backend
                    decode_one_token_args["backend"] = backend
                self.prefill = torch.compile(self.model, **prefil_args)
                self.decode_one_token = torch.compile(self.model, **decode_one_token_args)
            else:
                self.prefill = None
                self.decode_one_token = None

    def get_openvino_backend_options(self) -> Dict:
        return {
            "device": self._device,
            "model_caching": False,
            "cache_dir": "/home/susloval/work/tmp/model_cache",
        }

    def get_prefill(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        if (
            self.backend is not None
            and self.backend == "openvino"
            and (
                self.prefill is None
                or self._cached_prefill_input_ids.shape != input_ids.shape
                or self._cached_cache_position.shape != cache_position.shape
            )
        ):
            self._cached_prefill_input_ids = input_ids
            self._cached_cache_position = cache_position
            self.prefill = openvino.torch.backend.openvino(
                self.model, (input_ids, cache_position), options=self.get_openvino_backend_options()
            )
        return self.prefill

    def get_decode_one_token(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        if self.backend is not None and self.backend == "openvino" and self.decode_one_token is None:
            self.decode_one_token = openvino.torch.backend.openvino(
                self.model, (input_ids, cache_position), options=self.get_openvino_backend_options()
            )
        return self.decode_one_token

    def infer_prefill(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        if self.compile:
            _ = self.get_prefill(input_ids, cache_position)(input_ids, cache_position)
        else:
            self.model(input_ids, cache_position)

    def infer_decode_one_token(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        if self.compile:
            _ = self.get_decode_one_token(input_ids, cache_position)(input_ids, cache_position)
        else:
            self.model(input_ids, cache_position)

    @property
    def device(self) -> torch.device:
        return torch.device(self._device.lower())

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        cache_position = kwargs["cache_position"]
        past_len = cache_position[0]
        if past_len < input_ids.shape[1]:
            input_ids = input_ids[:, past_len:]

        return {"input_ids": input_ids, "cache_position": cache_position}

    def _save_pretrained(self, save_directory):
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if self.compile:
            if input_ids.shape[1] == 1:
                logits = self.get_decode_one_token(input_ids, cache_position)(input_ids, cache_position)
            else:
                logits = self.get_prefill(input_ids, cache_position)(input_ids, cache_position)
        else:
            logits = self.model(input_ids, cache_position)

        return CausalLMOutputWithPast(logits=logits)

    def can_generate(self):
        return True

    def _supports_default_dynamic_cache(self) -> bool:
        return False


class TorchExportableModuleWithStaticCacheDynamicShape(TorchExportableModuleWithStaticCache):
    def forward(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        outs = self.model(
            input_ids=input_ids,
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position,
            past_key_values=self.static_cache,
            use_cache=True,
        )
        return outs.logits


def convert_and_export_with_cache(model: PreTrainedModel, use_torch_export=True):
    """
    Convert a `PreTrainedModel` into an exportable module and export it using `torch.export`
    or `torch._export.capture_pre_autograd_graph`.
    """
    import torch.export._trace

    with torch.no_grad():
        example_input_ids = torch.ones(1, 8, dtype=torch.long)
        example_cache_position = torch.arange(0, 8, dtype=torch.long)
        model(example_input_ids)
        sequence_length = torch.export.Dim("sequence_length", min=1, max=128)
        dynamic_shapes = {"input_ids": {1: sequence_length}, "cache_position": {0: sequence_length}}

        if use_torch_export:
            exported_program = torch.export._trace._export(
                TorchExportableModuleWithStaticCacheDynamicShape(model),
                args=(example_input_ids, example_cache_position),
                pre_dispatch=False,
                strict=True,
                dynamic_shapes=dynamic_shapes,
            )
        else:
            exported_program = torch._export.capture_pre_autograd_graph(
                TorchExportableModuleWithStaticCacheDynamicShape(model),
                args=(example_input_ids, example_cache_position),
                dynamic_shapes=dynamic_shapes,
            )
        return exported_program


# SETTINGS
# model_id = "meta-llama/Meta-Llama-3-8B"
model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model_path = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"
# model_path = "/media/susloval/work/models/llms/pytorch/llama-3-8b/pytorch"
# exported_model_path = "/home/susloval/work/projects/nncf/experiments/torch_fx/model/Meta_Llama_3_8B"
exported_model_path = "/home/susloval/work/projects/nncf/experiments/torch_fx/model/TinyLlama_1.1B_Chat_v1.0"
fx_model_path = f"{exported_model_path}/exported_model.pt2"
device = "cpu"
use_torch_export = True
compile = True
# compile = False
backend = "openvino"
# backend = None
perf_bench = True
num_bench = 10
loading_from_file = False
compress_weights = False
# SETTINGS

with disable_patching():
    with torch.no_grad():
        if os.path.isfile(fx_model_path) and use_torch_export and loading_from_file:
            print("loading...")
            model_config = PretrainedConfig.from_pretrained(exported_model_path)
            generation_config = GenerationConfig.from_pretrained(exported_model_path)
            exported_model = torch.export.load(fx_model_path)
        else:
            print("loading...")
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.float32)
            model.generation_config.cache_implementation = "static"
            model.generation_config.cache_config = StaticCacheConfig(
                batch_size=1, max_cache_len=model.generation_config.max_length, device=device
            )
            model = model.eval()

            model_config = model.config
            generation_config = model.generation_config

            print("exporting...")
            exported_model = convert_and_export_with_cache(model, use_torch_export)
            if use_torch_export and loading_from_file:
                print("saving...")
                model_config.save_pretrained(exported_model_path)
                generation_config.save_pretrained(exported_model_path)
                torch.export.save(exported_model, fx_model_path)
        print("done")

        graph_module = exported_model.module() if use_torch_export else exported_model

        from experiments.torch_fx.constant_folding import constant_fold

        if compress_weights:
            constant_fold(graph_module)
            # weight compression
            graph_module = nncf.compress_weights(graph_module, mode=nncf.CompressWeightsMode.INT4_ASYM)

        fx_model = FXAutoModelForCausalLM(graph_module, model_config, compile=compile, backend=backend)

        if perf_bench:
            prefill_input_ids = torch.ones(1, 8, dtype=torch.long)
            prefill_cache_position = torch.arange(0, 8, dtype=torch.long)
            decode_one_token_input_ids = torch.tensor([[1]], dtype=torch.long)
            decode_one_token_cache_position = torch.tensor([8], dtype=torch.long)

            perfcounter.perf_start("first_infer_prefill")
            fx_model.infer_prefill(prefill_input_ids, prefill_cache_position)
            perfcounter.perf_end("first_infer_prefill")

            perfcounter.perf_start("first_infer_decode_one_token")
            fx_model.infer_decode_one_token(decode_one_token_input_ids, decode_one_token_cache_position)
            perfcounter.perf_end("first_infer_decode_one_token")

            for _ in range(num_bench):
                perfcounter.perf_start("infer_prefill")
                fx_model.infer_prefill(prefill_input_ids, prefill_cache_position)
                perfcounter.perf_end("infer_prefill")

            for _ in range(num_bench):
                perfcounter.perf_start("infer_decode_one_token")
                fx_model.infer_decode_one_token(decode_one_token_input_ids, decode_one_token_cache_position)
                perfcounter.perf_end("infer_decode_one_token")

        text = "Hey how are you doing today?"
        tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=device)
        token = tokenizer(text, return_tensors="pt")

        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.pad_token_id = tokenizer.eos_token_id
        generation_config.max_new_tokens = 15
        generation_config.repetition_penalty = 1.5

        perfcounter.perf_start("generate")
        generated_sequence = fx_model.generate(token["input_ids"], generation_config=generation_config)
        perfcounter.perf_end("generate")

        text = tokenizer.batch_decode(
            generated_sequence,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        print(text)
        print(perfcounter.perf_report())

exit()
