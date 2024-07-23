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

from typing import Optional

import torch
import torch.fx

import nncf
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.torch.model_graph_manager import OPERATORS_WITH_BIAS_METATYPES
from nncf.torch.model_graph_manager import find_const_node_in_constant_subgraph


def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
    if node.metatype not in OPERATORS_WITH_BIAS_METATYPES:
        return False
    return True


def get_const_data(node: NNCFNode, model: torch.fx.GraphModule) -> torch.Tensor:
    """
    Retrieves a constant tensor associated with a given node.

    :param const_node: The node associated with const data.
    :param model: The NNCFNetwork object.
    :return: A torch.Tensor object containing the constant value.
    """
    assert node.op == "get_attr"
    attr_itr = model
    if not hasattr(attr_itr, node.target):
        raise RuntimeError(f"Node referenced nonexistent target {node.target}")
    attr_itr = getattr(attr_itr, node.target)
    return torch.flatten(attr_itr.data)


def get_const_node(node: NNCFNode, port_id: int, graph: NNCFGraph) -> Optional[NNCFNode]:
    """
    Retrieves the constant node providing the input to a specific port of a given node in the NNCF graph.

    :param node: The NNCF node for which to find the constant input node.
    :param port_id: The ID of the input port to consider.
    :param graph: The NNCF graph containing the nodes.
    :return: The NNCF node providing the constant input to the specified port, or None if no such node is found.
    """
    for prev_node in graph.get_previous_nodes(node):
        edge = graph.get_edge(prev_node, node)
        if edge.input_port_id == port_id:
            weight_node = find_const_node_in_constant_subgraph(prev_node, graph)
            if weight_node is None:
                raise nncf.InternalError("Could not find a constant node in the model graph.")
            return weight_node


def get_const_data_on_port(node: NNCFNode, port_id: int, model: torch.fx.GraphModule) -> torch.Tensor:
    """
    Retrieves a constant tensor associated with a given node and input port in an NNCF graph.

    :param node: The node to retrieve the constant from.
    :param port_id:  The port id within the node that holds the constant.
    :param model: The NNCFNetwork object.
    :return: A torch.Tensor object containing the constant value, or None if the constant is not found.
    """
    graph = NNCFGraphFactory.create(model)
    const_node = get_const_node(node, port_id, graph)
    if const_node is None:
        return None

    return get_const_data(const_node, model)


def get_bias_value(node: NNCFNode, model: torch.fx.GraphModule) -> Optional[torch.Tensor]:
    """
    Returns the bias tensor for the node or for potential fused node.

    :param node: The node that corresponds to the operation with bias.
    :param model: The model that contains this operation.
    :return: The bias value that is applied to the output tensor of the node's operation.
    """
    from nncf.experimental.torch.fx.model_transformer import FXModelTransformer

    nncf_graph = NNCFGraphFactory.create(model)
    bias_node = nncf_graph.get_next_nodes(node)[0]
    graph_bias_node = FXModelTransformer.get_graph_node_by_name(model.graph, bias_node.node_name)
    return get_const_data(graph_bias_node.all_input_nodes[1], model)
