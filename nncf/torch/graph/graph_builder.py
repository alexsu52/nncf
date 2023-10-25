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
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Set

import torch

from nncf.common.graph.operator_metatypes import CONST_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import INPUT_NOOP_METATYPES
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.dynamic_graph.graph import DynamicGraph
from nncf.torch.dynamic_graph.graph_tracer import GraphTracer
from nncf.torch.dynamic_graph.layer_attributes_handlers import set_nodes_attributes_in_nncf_graph
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PT_OPERATOR_METATYPES


class GraphBuilder:
    def __init__(self, custom_forward_fn: Callable[[torch.nn.Module], Any]):
        self.custom_forward_fn = custom_forward_fn

    def build_dynamic_graph(
        self,
        model: torch.nn.Module,
        context_to_use: Optional[TracingContext] = None,
        as_eval: bool = False,
        trace_parameters: bool = False,
    ) -> DynamicGraph:
        tracer = GraphTracer(self.custom_forward_fn)
        return tracer.trace_graph(model, context_to_use, as_eval, trace_parameters)

    def build_graph(
        self,
        model: torch.nn.Module,
        context_to_use: Optional[TracingContext] = None,
        as_eval: bool = False,
        trace_parameters: bool = False,
    ) -> PTNNCFGraph:
        dynamic_graph = self.build_dynamic_graph(model, context_to_use, as_eval, trace_parameters)
        return GraphConverter.convert(dynamic_graph)


class GraphConverter:
    @staticmethod
    def convert(dynamic_graph: DynamicGraph) -> PTNNCFGraph:
        module_id_vs_known_op_addrs_map: Dict[int, Set[Scope]] = defaultdict(set)
        for dynamic_graph_node in dynamic_graph.get_all_nodes():
            module_id_vs_known_op_addrs_map[dynamic_graph_node.calling_module_id].add(
                dynamic_graph_node.op_exec_context.op_address
            )

        module_id_vs_sorted_scopes_map = {
            k: list(sorted([s.scope_in_model for s in v], key=str)) for k, v in module_id_vs_known_op_addrs_map.items()
        }

        nncf_graph = PTNNCFGraph()
        for dynamic_graph_node in dynamic_graph.get_all_nodes():
            op_address = dynamic_graph_node.op_exec_context.op_address

            metatype = PT_OPERATOR_METATYPES.get_operator_metatype_by_op_name(op_address.operator_name)
            if metatype.get_subtypes():
                subtype = metatype.determine_subtype(
                    dynamic_graph_node.layer_attributes, functions_kwargs=dynamic_graph_node.__dict__
                )
            else:
                subtype = None
            if subtype is not None:
                metatype = subtype

            is_integer_input = False
            if metatype in INPUT_NOOP_METATYPES:
                is_integer_input = dynamic_graph.is_integer_input_node(dynamic_graph_node)

            is_shared = len(module_id_vs_sorted_scopes_map[dynamic_graph_node.calling_module_id]) > 1
            canonical_scope = module_id_vs_sorted_scopes_map[dynamic_graph_node.calling_module_id][0]

            node_name = str(op_address)
            if metatype in CONST_NOOP_METATYPES:
                node_name = dynamic_graph_node.layer_attributes.name

            nncf_graph.add_nncf_node(
                node_name=node_name,
                node_type=op_address.operator_name,
                node_metatype=metatype,
                layer_attributes=dynamic_graph_node.layer_attributes,
                node_id_override=dynamic_graph_node.node_id,
                layer_name=str(canonical_scope),
                ignored_algorithms=dynamic_graph_node.ignored_algorithms,
                is_in_iteration_scope=dynamic_graph_node.is_in_iteration_scope,
                is_integer_input=is_integer_input,
                is_shared=is_shared,
            )

        for dynamic_graph_edge in dynamic_graph.get_all_edges():
            nncf_graph.add_edge_between_nncf_nodes(
                from_node_id=dynamic_graph_edge.from_node_id,
                to_node_id=dynamic_graph_edge.to_node_id,
                tensor_shape=dynamic_graph_edge.activation_shape,
                input_port_id=dynamic_graph_edge.input_port_id,
                output_port_id=dynamic_graph_edge.output_port_id,
                dtype=dynamic_graph_edge.dtype,
                parallel_input_port_ids=dynamic_graph_edge.parallel_input_port_ids,
            )

        set_nodes_attributes_in_nncf_graph(nncf_graph)
        return nncf_graph
