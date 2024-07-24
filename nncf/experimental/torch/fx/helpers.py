# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.fx
from torch.ao.quantization.pt2e.utils import fold_bn_weights_into_conv_node


def is_supported_batch_norm_for_training(node: torch.fx.Node):
    """
    Return True if the given node refers to an aten batch norm op QAT supports.
    """
    supported_ops = [
        torch.ops.aten._native_batch_norm_legit.default,
        torch.ops.aten.cudnn_batch_norm.default,
        torch.ops.aten.miopen_batch_norm.default,
    ]
    return node.target in supported_ops


def is_conv_node(node: torch.fx.Node):
    """
    Return whether the node refers to an aten conv op.
    """
    return node.op == "call_function" and node.target in [
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv2d.default,
    ]


def is_bn_node(node: torch.fx.Node):
    return (
        is_supported_batch_norm_for_training(node)
        or node.target == torch.ops.aten._native_batch_norm_legit_no_training.default
    )


def fuse_conv_bn(model: torch.fx.GraphModule) -> None:
    """
    BatchNorm operations have 3 output ports, to make it easier for alorithms to work with
    the target graph BatchNorm operations are being fused

    :param model: Model to apply transformations to.
    """
    has_bn = any(is_bn_node(node) for node in model.graph.nodes)
    if not has_bn:
        return

    for node in model.graph.nodes:
        if node.op != "call_function" or not is_bn_node(node):
            continue
        bn_node = node

        node = bn_node.args[0]
        if not is_conv_node(node):
            continue
        conv_node = node
        conv_weight_node = conv_node.args[1]
        conv_bias_node = conv_node.args[2] if len(conv_node.args) > 2 else None
        fold_bn_weights_into_conv_node(conv_node, conv_weight_node, conv_bias_node, bn_node, model)

    model.graph.eliminate_dead_code()
    model.recompile()
