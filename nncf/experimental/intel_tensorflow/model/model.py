"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import copy
import os
from nncf.experimental.intel_tensorflow.utils.utility import LazyImport
from nncf.experimental.intel_tensorflow.utils import logger
from nncf.experimental.intel_tensorflow.model.base_model import BaseModel

tf = LazyImport('tensorflow')

tensor_to_node = lambda s: list(set([x.split(':')[0] for x in s]))


def get_model_type(model):
    """Get mode type

    Args:
        model (string or model object): model path or model object

    Returns:
        type (string): model type
    """

    from nncf.experimental.intel_tensorflow.graph_editor.util import is_saved_model_format, is_ckpt_format
    if isinstance(model, tf.Graph):
        return 'graph'
    elif isinstance(model, tf.compat.v1.GraphDef):
        return 'graph_def'
    elif isinstance(model, tf.keras.Model):
        return 'keras'
    elif isinstance(model, tf.compat.v1.estimator.Estimator):
        return 'estimator'
    elif isinstance(model, str):
        model = os.path.abspath(os.path.expanduser(model))
        if (model.endswith('.pb') and os.path.isfile(model)):
            if is_saved_model_format(os.path.dirname(model)):
                # Warning: TF compatibility issue to load saved model. TF 2.3 keras.load
                # can load saved model from TF backend, but TF 2.4 cannot.
                try:
                    if tf.version.VERSION < '2.3.0':
                        logger.warn("keras model running on tensorflow 2.2.0 and"
                                    " lower may have problem.")
                    model = tf.keras.models.load_model(model)
                    if isinstance(model, tf.keras.Model):
                        return 'keras'
                    else:
                        return 'saved_model'
                except:
                    # can't use keras load
                    return 'saved_model'
            else:
                return 'frozen_pb'
        elif model.endswith('.ckpt') and os.path.isfile(model):
            return 'slim'
        elif os.path.isdir(model):
            if is_ckpt_format(model):
                return 'checkpoint'
            elif is_saved_model_format(model):
                # it's very ugly tf version issue, in tf2.3 keras.load can
                #batch_size_(batch_size), load saved model from tf backend, but tf2.4 it will crash
                try:
                    if tf.version.VERSION < '2.3.0':
                        logger.warn("keras model running on tensorflow 2.2.0 and"
                                    " lower may have problem.")
                    model = tf.keras.models.load_model(model)
                    if isinstance(model, tf.keras.Model):
                        return 'keras'
                    else:
                        return 'saved_model'
                except:
                    # can't use keras load
                    return 'saved_model'
        elif os.path.isfile(model + '.pb'):
            return 'frozen_pb'

    raise ValueError('model {} has not recognized model type....'.format(model))


def validate_graph_node(graph_def, node_names):
    """Validate nodes exist in the graph_def

    Args:
        graph_def (tf.compat.v1.GraphDef): tf.compat.v1.GraphDef object
        node_names (list of string): node names to be validated
    """

    if len(node_names) == 0:
        return False
    all_node_name = [node.name for node in graph_def.node]
    for user_name in node_names:
        if user_name not in all_node_name:
            logger.warn(
                str("Node name {} specified in yaml doesn't exist in the model.").
                format(user_name))
            return False
    return True


def validate_and_inference_input_output(graph_def, \
    input_tensor_names, output_tensor_names):
    """validate and inference the input and output tensor names of graph_def

    Args:
        graph_def (tf.compat.v1.GraphDef): tf.compat.v1.GraphDef object
        input_tensor_names (list of string): input_tensor_names of graph_def
        output_tensor_names (list of string): output_tensor_names of graph_def

    Returns:
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """
    from nncf.experimental.intel_tensorflow.graph_editor.util import get_input_output_node_names
    temp_output_tensor_names = []
    if validate_graph_node(graph_def, tensor_to_node(input_tensor_names)):
        input_tensor_names = input_tensor_names
    else:
        input_tensor_names, temp_output_tensor_names = get_input_output_node_names(graph_def)

    if validate_graph_node(graph_def, tensor_to_node(output_tensor_names)):
        output_tensor_names = output_tensor_names
    elif temp_output_tensor_names:
        output_tensor_names = temp_output_tensor_names
    else:
        _, output_tensor_names = get_input_output_node_names(graph_def)

    return input_tensor_names, output_tensor_names


def graph_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with tf.compat.v1.Graph

    Args:
        model (tf.compat.v1.Graph): tf.compat.v1.Graph object
        input_tensor_names (list of string): input_tensor_names of model
        output_tensor_names (list of string): output_tensor_names of model

     Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """

    config = tf.compat.v1.ConfigProto()
    config.use_per_session_threads = 1
    config.inter_op_parallelism_threads = 1
    sess = tf.compat.v1.Session(graph=model, config=config)

    input_tensor_names, output_tensor_names = validate_and_inference_input_output(\
        model.as_graph_def(), input_tensor_names, output_tensor_names)

    return sess, input_tensor_names, output_tensor_names


def graph_def_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with tf.compat.v1.GraphDef

    Args:
        model (tf.compat.v1.GraphDef): tf.compat.v1.GraphDef object
        input_tensor_names (list of string): input_tensor_names of model
        output_tensor_names (list of string): output_tensor_names of model

     Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """

    graph = tf.Graph()
    try:
        with graph.as_default():
            tf.import_graph_def(model, name='')
    except:
        input_tensor_names, output_tensor_names = validate_and_inference_input_output(\
            model, input_tensor_names, output_tensor_names)
        from nncf.experimental.intel_tensorflow.graph_editor.util import fix_ref_type_of_graph_def
        from nncf.experimental.intel_tensorflow.graph_editor.util import strip_unused_nodes
        model = fix_ref_type_of_graph_def(model)
        input_node_names = tensor_to_node(input_tensor_names)
        output_node_names = tensor_to_node(output_tensor_names)
        model = strip_unused_nodes(model, input_node_names, output_node_names)
        with graph.as_default():
            tf.import_graph_def(model, name='')

    return graph_session(graph, input_tensor_names, output_tensor_names, **kwargs)


def frozen_pb_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with frozen pb

    Args:
        model (string): model path
        input_tensor_names (list of string): input_tensor_names of model
        output_tensor_names (list of string): output_tensor_names of model

     Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """

    graph_def = tf.compat.v1.GraphDef()
    model = model if model.endswith('.pb') else model + '.pb'
    with open(model, 'rb') as f:
        graph_def.ParseFromString(f.read())
    return graph_def_session(graph_def, input_tensor_names, \
        output_tensor_names, **kwargs)


def keras_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with keras model

    Args:
        model (string or tf.keras.Model): model path or tf.keras.Model object
        input_tensor_names (list of string): input_tensor_names of model
        output_tensor_names (list of string): output_tensor_names of model

     Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """

    assert tf.version.VERSION >= '2.3.0', 'keras model need tensorflow version >= 2.3.0....'
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    if not isinstance(model, tf.keras.Model):
        model = tf.keras.models.load_model(model)
    kwargs = dict(zip(model.input_names, model.inputs))
    if tf.version.VERSION > '2.2.0' and tf.version.VERSION < '2.5.0':
        from tensorflow.python.keras.engine import keras_tensor
        if keras_tensor.keras_tensors_enabled():
            for name, tensor in kwargs.items():
                kwargs[name] = tensor.type_spec
    elif tf.version.VERSION >= '2.5.0':
        for name, tensor in kwargs.items():
            kwargs[name] = tensor.type_spec
    full_model = tf.function(lambda **kwargs: model(kwargs.values()))
    concrete_function = full_model.get_concrete_function(**kwargs)
    frozen_model = convert_variables_to_constants_v2(concrete_function)

    from tensorflow.python.training import saver
    from tensorflow.core.protobuf import config_pb2
    from tensorflow.python.grappler import tf_optimizer
    from tensorflow.core.protobuf import meta_graph_pb2
    graph_def = frozen_model.graph.as_graph_def()
    input_names = [node.name for node in graph_def.node if node.op == 'Placeholder']
    output_names = [output.split(':')[0] for output in model.output_names]
    # replace the output name with squential
    for output_name in output_names:
        for node in graph_def.node[::-1]:
            if node.op == 'Identity' and output_name in node.input[0]:
                node.name = output_name
                break

    grappler_meta_graph_def = saver.export_meta_graph(
        graph_def=graph_def, graph=frozen_model.graph)

    # Add a collection 'train_op' so that Grappler knows the outputs.
    fetch_collection = meta_graph_pb2.CollectionDef()
    for array in model.output_names:
      fetch_collection.node_list.value.append(array)
    grappler_meta_graph_def.collection_def["train_op"].CopyFrom(
        fetch_collection)
    grappler_session_config = config_pb2.ConfigProto()
    rewrite_options = grappler_session_config.graph_options.rewrite_options
    rewrite_options.optimizers.append('constfold')
    rewrite_options.min_graph_nodes = -1
    graph_def = tf_optimizer.OptimizeGraph(grappler_session_config, \
                        grappler_meta_graph_def, graph_id=b"tf_graph")

    return graph_def_session(graph_def, input_names, output_names, **kwargs)


def slim_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with slim model

    Args:
        model (string): model path
        input_tensor_names (list of string): input_tensor_names of model
        output_tensor_names (list of string): output_tensor_names of model

     Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """

    assert tf.version.VERSION < '2.0.0', 'slim model only used in tensorflow 1.x'
    from .nets_factory import TFSlimNetsFactory
    factory = TFSlimNetsFactory()
    assert 'name' in kwargs, 'model name should be set in slim checkpoint....'
    assert kwargs['name'] in factory.default_slim_models, \
        'only support topology {}'.format(factory.default_slim_models)
    net = copy.deepcopy(factory.networks_map[kwargs['name']])
    model_func = net.pop('model')
    arg_scope = net.pop('arg_scope')()
    inputs_shape = net.pop('input_shape')
    kwargs = net
    import tf_slim as slim
    with tf.Graph().as_default():
        images = tf.compat.v1.placeholder(name='input', dtype=tf.float32, \
            shape=inputs_shape)
        with tf.compat.v1.Session() as sess:
            with slim.arg_scope(arg_scope) as scope:  # pylint: disable=not-context-manager
                model_func(images, is_training=False, **kwargs)
            graph_def = sess.graph.as_graph_def()
            output_tensor_names = output_tensor_names if len(output_tensor_names) > 0 \
                else [graph_def.node[-1].name]

            from tensorflow.python.tools.freeze_graph import freeze_graph_with_def_protos
            graph_def = freeze_graph_with_def_protos(
                input_graph_def=graph_def,
                input_saver_def=None,
                input_checkpoint=model,
                output_node_names=','.join(output_tensor_names),
                restore_op_name='save/restore_all',
                filename_tensor_name='save/Const:0',
                output_graph='',
                clear_devices=True,
                initializer_nodes='')

    return graph_def_session(graph_def, ['input'], output_tensor_names)


def checkpoint_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with ckpt model

    Args:
        model (string): model path
        input_tensor_names (list of string): input_tensor_names of model
        output_tensor_names (list of string): validated output_tensor_names of model

     Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """

    assert output_tensor_names is not None and len(output_tensor_names) > 0, \
        'outputs should not be None of checkpoint....'

    ckpt_prefix = [os.path.splitext(i)[0] for i in os.listdir(model) \
        if i.endswith(".meta")][0]

    config = tf.compat.v1.ConfigProto()
    config.use_per_session_threads = 1
    config.inter_op_parallelism_threads = 1
    graph = tf.Graph()
    sess = tf.compat.v1.Session(graph=graph, config=config)
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(\
            os.path.join(model, ckpt_prefix + '.meta'), clear_devices=True)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, os.path.join(model, ckpt_prefix))

    from nncf.experimental.intel_tensorflow.graph_editor.util import get_input_output_node_names
    if validate_graph_node(sess.graph.as_graph_def(), tensor_to_node(input_tensor_names)):
        input_tensor_names = input_tensor_names
    else:
        input_tensor_names, _ = get_input_output_node_names(sess.graph.as_graph_def())
    return sess, input_tensor_names, output_tensor_names


def estimator_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with estimator model

    Args:
        model (tf.estimator.Estimator): tf.estimator.Estimator object
        input_tensor_names (list of string): input_tensor_names of model
        output_tensor_names (list of string): output_tensor_names of model
        kwargs (dict): other required parameters, like input_fn

     Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """

    assert 'input_fn' in kwargs, 'input func should be supplied for estimator session....'
    with tf.Graph().as_default() as g:
      features, input_hooks = model._get_features_from_input_fn(
          kwargs['input_fn'], tf.estimator.ModeKeys.PREDICT)
      estimator_spec = model._call_model_fn(features, None,
          tf.estimator.ModeKeys.PREDICT, model.config)

      if len(output_tensor_names) == 0:
          outputs = [tensor.name for tensor in estimator_spec.predictions.values()] if\
              isinstance(estimator_spec.predictions, dict) else \
                  [estimator_spec.predictions.name]
      else:
          outputs = output_tensor_names

      logger.info("Estimator output tensor names are {}.".format(outputs))
      with tf.compat.v1.Session(graph=g) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # Freezing a graph requires output_node_names, which can be found in
        # estimator_spec.predictions that contains prediction tensors as a
        # dictionary
        # When a model uses Iterator, we need to have 'MakeIterator' (default
        # name used by TF) in the output_node_names as well.
        output_nodes = list(set([output.split(':')[0] for output in outputs]))
        if 'MakeIterator' in [node.op for node in g.as_graph_def().node]:
            output_nodes.append('MakeIterator')

        graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
          g.as_graph_def(), output_nodes)

      return graph_def_session(graph_def, input_tensor_names, outputs)


def saved_model_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with saved model

    Args:
        model (string): model path
        input_tensor_names (list of string): input_tensor_names of model
        output_tensor_names (list of string): output_tensor_names of model

     Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """
    config = tf.compat.v1.ConfigProto()
    config.use_per_session_threads = 1
    config.inter_op_parallelism_threads = 1
    sess = tf.compat.v1.Session(graph=tf.Graph(), config=config)
    loader = tf.compat.v1.saved_model.loader.load(sess, ["serve"], model)
    if len(input_tensor_names) == 0:
        input_tensor_names = [i.name for _, i in \
            loader.signature_def['serving_default'].inputs.items()]
    else:
        assert validate_graph_node(\
            sess.graph.as_graph_def(), tensor_to_node(input_tensor_names)), \
                'tensor names {} not in the graph'.format(input_tensor_names)

    if len(output_tensor_names) == 0:
        output_tensor_names = [i.name for _, i in \
            loader.signature_def['serving_default'].outputs.items()]
    else:
        assert validate_graph_node(\
            sess.graph.as_graph_def(), tensor_to_node(output_tensor_names)), \
                'tensor names {} not in the graph'.format(output_tensor_names)

    return sess, input_tensor_names, output_tensor_names


# it's necessary that a session with input output tensors to run the model
SESSIONS = {'frozen_pb': frozen_pb_session,
            'graph_def': graph_def_session,
            'graph': graph_session,
            'saved_model': saved_model_session,
            'keras': keras_session,
            'checkpoint': checkpoint_session,
            'estimator': estimator_session,
            'slim': slim_session,}


class TensorflowBaseModel(BaseModel):
    """Build TensorflowBaseModel object

    Args:
        model (string or tensorflow model object): model path or model object
        kwargs (dict): other required parameters, like input_fn

    """

    def __init__(self, model, **kwargs):

        self._model = model
        self._name = ''
        self.kwargs = kwargs
        self._graph_info = {}
        self._input_tensor_names = []
        self._output_tensor_names = []
        self._model_type = ''
        self._sess = None
        self._iter_op = None
        self._workspace_path = ''
        self._q_config = None

    def framework(self):
        return 'tensorflow'

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self.kwargs.update({'name': name})
        self._name = name
        
    @property
    def q_config(self):
        return self._q_config

    @q_config.setter
    def q_config(self, q_config):
        self._q_config = q_config

    @property
    def workspace_path(self):
        return self._workspace_path

    @workspace_path.setter
    def workspace_path(self, path):
        self._workspace_path = path

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, model_type):
        assert model_type in SESSIONS, 'model type not supported....'
        self._model_type = model_type

    @property
    def model(self):
        return self.graph

    @property
    def graph_def(self):
        return self.graph.as_graph_def()

    @property
    def graph_info(self):
        self._graph_info = {}
        for node in self.graph_def.node:
            self._graph_info[node.name] = node.op
        return self._graph_info

    @property
    def sess(self):
        if self._sess is None:
            self._load_sess(self._model, **self.kwargs)
        return self._sess

    @property
    def graph(self):
        return self.sess.graph

    @graph_def.setter
    def graph_def(self, graph_def):
        if self._sess is not None:
            self._sess.close()
        output_sess =  SESSIONS['graph_def'](graph_def,\
                                             self._input_tensor_names, \
                                             self._output_tensor_names)

        self._sess = output_sess[0]
        self._input_tensor_names = output_sess[1]
        self._output_tensor_names = output_sess[2]
        self.model_type = 'graph_def'

    def _load_sess(self, model, **kwargs):
        if self.name:
            kwargs.update({'name': self.name})
        # assert self.model_type, 'model type not set....'
        output_sess = SESSIONS[self.model_type](model,
                                                self._input_tensor_names, \
                                                self._output_tensor_names,
                                                **kwargs)
        self._sess = output_sess[0]
        self._input_tensor_names = output_sess[1]
        self._output_tensor_names = output_sess[2]

        tf.compat.v1.get_variable_scope().reuse_variables()
        return self._sess

    @property
    def iter_op(self):
        if self._sess is None:
            self._load_sess(self._model, **self.kwargs)
        op_list = [node.op for node in self._sess.graph.as_graph_def().node]
        if 'MakeIterator' in op_list:
            self._iter_op = self._sess.graph.get_operation_by_name(\
                'MakeIterator')
        return self._iter_op

    @property
    def input_tensor_names(self):
        if len(self._input_tensor_names) == 0:
            self._load_sess(self._model, **self.kwargs)
        return copy.deepcopy(self._input_tensor_names)

    @input_tensor_names.setter
    def input_tensor_names(self, tensor_names):
        if len(tensor_names) == 0:
            logger.warn("Input tensor names should not be empty.")
            return
        if self._sess is not None:
            assert validate_graph_node(\
                self.graph_def, tensor_to_node(tensor_names)), \
                'tensor names {} not in graph'.format(tensor_names)
        self._input_tensor_names = tensor_names

    @property
    def output_tensor_names(self):
        if len(self._output_tensor_names) == 0:
            self._load_sess(self._model, **self.kwargs)
        return copy.deepcopy(self._output_tensor_names)

    @output_tensor_names.setter
    def output_tensor_names(self, tensor_names):
        if len(tensor_names) == 0:
            logger.warn("Output tensor names should not be empty.")
            return
        if self._sess is not None:
            assert validate_graph_node(\
                self.graph_def, tensor_to_node(tensor_names)), \
                'tensor names {} not in graph'.format(tensor_names)
        self._output_tensor_names = tensor_names

    # input/output node names and input/output tensor
    # come from input/output tensor names, so do not support assign these values
    @property
    def input_node_names(self):
        return copy.deepcopy(tensor_to_node(self.input_tensor_names))

    @property
    def output_node_names(self):
        output_node_names = tensor_to_node(self.output_tensor_names)
        if self.iter_op is not None:
            output_node_names.append('MakeIterator')
        return copy.deepcopy(output_node_names)

    @property
    def input_tensor(self):
        from nncf.experimental.intel_tensorflow.graph_editor.util import get_tensor_by_name
        return [get_tensor_by_name(\
            self.graph, x) for x in self.input_tensor_names]

    @property
    def output_tensor(self):
        from nncf.experimental.intel_tensorflow.graph_editor.util import get_tensor_by_name
        return [get_tensor_by_name(\
            self.graph, x) for x in self.output_tensor_names]

    def save(self, root):
        root = os.path.abspath(os.path.expanduser(root))
        # if not have suffix, default append .pb
        os.makedirs(os.path.dirname(root), exist_ok=True)
        pb_file = root if os.path.split(root)[-1].endswith('.pb') else root + '.pb'
        f = tf.io.gfile.GFile(pb_file, 'wb')
        f.write(self.graph_def.SerializeToString())
        logger.info("Save quantized model to {}.".format(pb_file))


class TensorflowSavedModelModel(TensorflowBaseModel):

    def save(self, root):
        root = os.path.abspath(os.path.expanduser(root))
        if os.path.exists(root):
            import shutil
            shutil.rmtree(root)

        os.makedirs(root, exist_ok=True)

        from tensorflow.python.saved_model import signature_constants
        from tensorflow.python.saved_model import tag_constants
        from nncf.experimental.intel_tensorflow.graph_editor.util import get_tensor_by_name
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(root)
        sigs = {}
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            #(TODO) not directly use self._sess.graph, use self.graph
            tf.import_graph_def(self.graph.as_graph_def(), name="")
            g = tf.compat.v1.get_default_graph()
            inp = [get_tensor_by_name(g, x) for x in self._input_tensor_names]
            out = [get_tensor_by_name(g, x) for x in self._output_tensor_names]
            sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
                {k: v for k, v in zip(self._input_tensor_names, inp)},
                {k: v for k, v in zip(self._output_tensor_names, out)})
            builder.add_meta_graph_and_variables(sess,
                                                 [tag_constants.SERVING],
                                                 signature_def_map=sigs)
        builder.save()
        logger.info("Save quantized model to {}.".format(root))


class TensorflowCheckpointModel(TensorflowBaseModel):

    @property
    def graph_def(self):
        if self.model_type == 'graph_def':
            return self.sess.graph.as_graph_def()
        from nncf.experimental.intel_tensorflow.graph_editor.util import _parse_ckpt_bn_input
        from tensorflow.python.framework import graph_util
        graph_def = self.sess.graph.as_graph_def()
        graph_def = _parse_ckpt_bn_input(graph_def)
        return graph_util.convert_variables_to_constants(
            sess=self._sess,
            input_graph_def=graph_def,
            output_node_names=self.output_node_names)

    @graph_def.setter
    def graph_def(self, graph_def):
        if self._sess is not None:
            self._sess.close()
        output_sess = SESSIONS['graph_def'](graph_def,
                                            self._input_tensor_names, \
                                            self._output_tensor_names)
        self._sess = output_sess[0]
        self._input_tensor_names = output_sess[1]
        self._output_tensor_names = output_sess[2]
        self.model_type = 'graph_def'


TENSORFLOW_MODELS = {'frozen_pb': TensorflowBaseModel,
                     'graph_def': TensorflowBaseModel,
                     'graph': TensorflowBaseModel,
                     'checkpoint': TensorflowCheckpointModel,
                     'estimator': TensorflowBaseModel,
                     'slim': TensorflowBaseModel,
                     'saved_model': TensorflowSavedModelModel,
                     'keras': TensorflowSavedModelModel,}


class TensorflowModel(object):

    def __new__(cls, model_type, root, **kwargs):
        model = TENSORFLOW_MODELS[model_type](root, **kwargs)
        model.model_type = model_type
        return model


class Model(object):
    """common Model just collect the infos to construct a Model
    """

    def __new__(cls, root, **kwargs):
        """Wrap raw framework model format or path with specific infos

        Args:
            root:   raw model format. For Tensorflow model, could be path to frozen pb file,
                    path to ckpt or savedmodel folder, loaded estimator/graph_def/graph/keras
                    model object. For PyTorch model, it's torch.nn.model instance.
                    For MXNet model, it's mxnet.symbol.Symbol or gluon.HybirdBlock instance.
            kwargs: specific model format will rely on extra infomation to build the model
                    a. estimator: need input_fn to initialize the Model, it will look like this
                                  when initialize an estimator model:
                                  model = Model(estimator_object, input_fn=estimator_input_fn)

        """
        model_type = get_model_type(root)
        model = TensorflowModel(model_type, root, **kwargs)
        return model
