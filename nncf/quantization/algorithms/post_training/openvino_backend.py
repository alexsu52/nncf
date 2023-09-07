# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import openvino.runtime as ov
from openvino.runtime import opset9 as opset
from tqdm import tqdm

from nncf import Dataset
from nncf.data.dataset import DataItem
from nncf.quantization.algorithms.post_training.backend import PostTrainingBackend


def _add_results(model: ov.Model, node: ov.Node) -> Tuple[ov.Model, List[str]]:
    extra_model_outputs = []
    result_names = []
    for input in node.inputs():
        output = input.get_source_output()
        output_name = output.get_node().get_friendly_name()
        result_name = f"{output_name}/if_output"

        result = opset.result(output, name=result_name)

        tensor = result.get_output_tensor(0)
        current_names = tensor.get_names()
        current_names.add(result_name)
        tensor.set_names(current_names)
        result_names.append(result_name)
        extra_model_outputs.append(result)
    return (
        ov.Model(
            results=extra_model_outputs,
            sinks=[op for op in model.get_ops() if op.get_type_name() == "Assign"],
            parameters=model.get_parameters(),
            name=model.friendly_name,
        ),
        result_names,
    )


class OVPostTrainingBackend(PostTrainingBackend):
    IF_OP_MODEL_INPUT_PORTS = (0, 1)

    @staticmethod
    def is_single_model(model: ov.Model) -> bool:
        for op in model.get_ops():
            if op.get_type_name() == "If":
                return False
        return True

    @staticmethod
    def get_child_models(model: ov.Model) -> List[Tuple[ov.Model, Dict[str, Any]]]:
        child_models = []
        for op in model.get_ops():
            if op.get_type_name() == "If":
                for port_id in OVPostTrainingBackend.IF_OP_MODEL_INPUT_PORTS:
                    input_indices = [desc.input_index for desc in op.get_input_descriptions(port_id)]
                    input_names = [op.input_values()[index].any_name for index in input_indices]
                    child_models.append(
                        (
                            op.get_function(port_id),
                            input_names,
                            {
                                "if_op": op,
                                "if_op_model_input_port_id": port_id,
                            },
                        )
                    )
        return child_models

    @staticmethod
    def add_additional_outputs(model: ov.Model) -> Tuple[ov.Model, List[str]]:
        for op in model.get_ops():
            if op.get_type_name() == "If":
                # TODO: fix adding multiple IF
                model_with_additional_results, result_names = _add_results(model, op)
        return model_with_additional_results

    @staticmethod
    def set_child_model(child_model: ov.Model, if_op: ov.Node, if_op_model_input_port_id: int) -> None:
        if_op.set_function(if_op_model_input_port_id, child_model)

    @staticmethod
    def dump_model(model: ov.Model, dir: str, if_op: ov.Node, if_op_model_input_port_id: int) -> None:
        name = if_op.get_friendly_name().replace("/", "")
        if if_op_model_input_port_id == 0:
            postfix = "then"
        if if_op_model_input_port_id == 1:
            postfix = "else"
        model_name = f"{name}_{postfix}.xml"
        model_path = Path(dir) / model_name
        ov.serialize(model, model_path)
