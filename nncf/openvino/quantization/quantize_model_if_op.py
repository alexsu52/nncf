from itertools import islice
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, TypeVar

import openvino.runtime as ov
from tqdm import tqdm

from nncf import Dataset
from nncf.common import factory
from nncf.common.engine import Engine
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.logging import nncf_logger
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.openvino.graph.metatypes.openvino_metatypes import OVIfMetatype
from nncf.openvino.graph.node_utils import has_if_op
from nncf.openvino.graph.transformations.commands import OVExtractIfSubgraphCommand
from nncf.openvino.graph.transformations.commands import OVOutputInsertionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.graph.transformations.commands import OVUpdateIfSubgraphCommand
from nncf.quantization.algorithms.algorithm import Algorithm


def _make_dataset_for_children_model(
    engine: Engine,
    calibration_dataset: Dataset,
    if_cond_input_name: str,
    then_model_input_names,
    else_model_input_names,
    subset_size: int,
) -> Tuple[Dataset, Dataset]:
    """
    Returns dataset for a child model.

    :param engine: Engine to infer parent model to obtain dataitems for a child dataset.
    :param calibration_dataset: Dataset to infer parent model.
    :param if_cond_input_name: Input name of If node condition.
    :param child_model_input_names: - Names of inputs for child model
    (should be in the order of passing them to a model).
    :param if_submodel_condition: If node submodel condition.
    :return Dataset: Dataset for child model.
    """
    then_dataset, else_dataset = [], []
    calibration_dataset_size = (
        min(subset_size, calibration_dataset.get_length())
        if calibration_dataset.get_length() is not None
        else subset_size
    )
    for input_data in tqdm(
        islice(calibration_dataset.get_inference_data(), calibration_dataset_size),
        total=calibration_dataset_size,
        desc="Collect dataset for then and else models:",
    ):
        data_item = []
        results = engine.infer(input_data)
        if results[if_cond_input_name]:
            for name in then_model_input_names:
                data_item.append(results[name])
            then_dataset.append(data_item)
        else:
            for name in else_model_input_names:
                data_item.append(results[name])
            else_dataset.append(data_item)
    nncf_logger.info(f"The length of dataset for then body is {len(then_dataset)}")
    nncf_logger.info(f"The length of dataset for else body is {len(else_dataset)}")
    return Dataset(then_dataset), Dataset(else_dataset)


def _extract_if_submodel(
    model_transformer: ModelTransformer, if_node: NNCFNode, if_submodel_condition: bool
) -> ov.Model:
    """
    Returns if submodel of If node laying on an input port if_submodel_port_id of If node.

    :param model_transformer: ModelTransformer instance.
    :param if_node: If node.
    :param if_submodel_condition: If True returns True submodel of If node, otherwise - False submodel.
    :return: If submodel.
    """
    transformation_layout = TransformationLayout()
    command = OVBackend.create_extract_if_subgraph_command(if_node, if_submodel_condition)
    transformation_layout.register(command)
    return model_transformer.transform(transformation_layout)


def _update_if_submodel(
    model_transformer: ModelTransformer, if_node: NNCFNode, if_submodel_condition: bool, submodel: ov.Model
) -> ov.Model:
    """
    Update submodel of If node.

    :param model_transformer: ModelTransformer instance.
    :param if_node: If node.
    :param if_submodel_condition: Condition of If node submodel.
    :param submodel: New submodel.
    :return: Updated model with a new submodel.
    """
    transformation_layout = TransformationLayout()
    command = OVBackend.create_update_subgraph_command(if_node, if_submodel_condition, submodel)
    transformation_layout.register(command)
    return model_transformer.transform(transformation_layout)


def _add_outputs_before_if_node(model_transformer: ModelTransformer, model: ov.Model, if_node: NNCFNode) -> ov.Model:
    """
    Inserts extra outputs on If node inputs.

    :param model_transformer: ModelTransformer instance.
    :param model: Model instance.
    :param if_node: If node.
    :return: Model with extra outputs before If node.
    """
    transformation_layout = TransformationLayout()
    for command in OVBackend.create_output_insertion_commands(model, if_node):
        transformation_layout.register(command)
    return model_transformer.transform(transformation_layout)


def apply_algorithm_if_bodies(
    algorithm: Algorithm,
    parent_model: ov.Model,
    parent_graph: NNCFGraph,
    parent_dataset: Dataset,
    subset_size: int,
    parent_statistic_points: Optional[StatisticPointsContainer] = None,
) -> ov.Model:
    """

    :param parent_model:
    :param parent_graph:
    :param parent_dataset:
    :param parent_statistic_points:
    :param parent_model_cnt:
    :return:
    """
    nncf_logger.info(f"Quantize a new submodel")
    quantized_model = algorithm.apply(parent_model, parent_graph, parent_statistic_points, parent_dataset)
    if not has_if_op(parent_model):
        return quantized_model
    model_transformer_fp32 = factory.ModelTransformerFactory.create(parent_model)
    for if_node in parent_graph.get_nodes_by_metatypes([OVBackend.if_node_metatype()]):
        then_model_input_names = OVBackend.get_if_subgraph_input_names(parent_model, if_node, True)
        else_model_input_names = OVBackend.get_if_subgraph_input_names(parent_model, if_node, False)
        if_cond_input_name = OVBackend.get_if_cond_input_name(parent_model, if_node)
        parent_model_with_additional_outputs = _add_outputs_before_if_node(
            model_transformer_fp32, parent_model, if_node
        )
        then_dataset, else_dataset = _make_dataset_for_children_model(
            factory.EngineFactory.create(parent_model_with_additional_outputs),
            parent_dataset,
            if_cond_input_name,
            then_model_input_names,
            else_model_input_names,
            subset_size,
        )
        then_model = _extract_if_submodel(model_transformer_fp32, if_node, True)
        else_model = _extract_if_submodel(model_transformer_fp32, if_node, False)
        then_quantized_model = apply_algorithm_if_bodies(
            algorithm, then_model, NNCFGraphFactory.create(then_model), then_dataset, subset_size, None
        )
        else_quantized_model = apply_algorithm_if_bodies(
            algorithm, else_model, NNCFGraphFactory.create(else_model), else_dataset, subset_size, None
        )
        model_transformer_int8 = factory.ModelTransformerFactory.create(quantized_model)
        quantized_model = _update_if_submodel(model_transformer_int8, if_node, True, then_quantized_model)
        model_transformer_int8 = factory.ModelTransformerFactory.create(quantized_model)
        quantized_model = _update_if_submodel(model_transformer_int8, if_node, False, else_quantized_model)
    return quantized_model


class OVBackend:
    @staticmethod
    def _get_if_submodel_port_id(if_submodel_condition: bool):
        return int(not if_submodel_condition)

    @staticmethod
    def if_node_metatype():
        return OVIfMetatype

    @staticmethod
    def get_if_subgraph_input_names(model: ov.Model, if_node: NNCFNode, if_submodel_condition: bool) -> List[str]:
        input_names = []
        name_to_node_mapping = {op.get_friendly_name(): op for op in model.get_ops()}
        ov_node = name_to_node_mapping[if_node.node_name]
        input_indices = [
            desc.input_index
            for desc in ov_node.get_input_descriptions(OVBackend._get_if_submodel_port_id(if_submodel_condition))
        ]
        input_names.extend([ov_node.input_values()[index].any_name for index in input_indices])
        return input_names

    @staticmethod
    def get_if_cond_input_name(model: ov.Model, if_node: NNCFNode) -> str:
        name_to_node_mapping = {op.get_friendly_name(): op for op in model.get_ops()}
        ov_node = name_to_node_mapping[if_node.node_name]
        return ov_node.input_values()[0].any_name

    @staticmethod
    def create_update_subgraph_command(
        if_node: NNCFNode, if_submodel_condition: bool, subgraph_model: ov.Model
    ) -> OVUpdateIfSubgraphCommand:
        target_point = OVTargetPoint(
            TargetType.LAYER, if_node.node_name, OVBackend._get_if_submodel_port_id(if_submodel_condition)
        )
        return OVUpdateIfSubgraphCommand(target_point, subgraph_model)

    @staticmethod
    def create_extract_if_subgraph_command(
        if_node: NNCFNode, if_submodel_condition: bool
    ) -> OVExtractIfSubgraphCommand:
        return OVExtractIfSubgraphCommand(if_node.node_name, if_submodel_condition)

    @staticmethod
    def create_output_insertion_commands(model: ov.Model, if_node: NNCFNode) -> List[OVOutputInsertionCommand]:
        commands = []
        name_to_node_mapping = {op.get_friendly_name(): op for op in model.get_ops()}
        ov_node = name_to_node_mapping[if_node.node_name]
        for port_id in range(len(ov_node.inputs())):
            commands.append(
                OVOutputInsertionCommand(OVTargetPoint(TargetType.PRE_LAYER_OPERATION, if_node.node_name, port_id))
            )
        return commands
