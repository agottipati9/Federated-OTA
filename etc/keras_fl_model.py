"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2020 All Rights Reserved.
"""

import logging
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import time
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import NumpyArrayIterator
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

from ibmfl.util import config
from ibmfl.util import fl_metrics
from ibmfl.model.fl_model import FLModel
from ibmfl.model.model_update import ModelUpdate
from ibmfl.exceptions import FLException, LocalTrainingException, ModelException

logger = logging.getLogger(__name__)


class KerasFLModel(FLModel):
    """
    Wrapper class for importing keras and tensorflow.keras models.
    """

    def __init__(self, model_name,
                 model_spec,
                 keras_model=None,
                 **kwargs):
        """
        Create a `KerasFLModel` instance from a Keras model.
        If keras_model is provided, it will use it; otherwise it will take
        the model_spec to create the model.
        Assumes the `model` passed as argument is compiled.

        :param model_name: String specifying the type of model e.g., Keras_CNN
        :type model_name: `str`
        :param model_spec: Specification of the keras_model
        :type model_spec: `dict`
        :param keras_model: Compiled keras model.
        :type keras_model: `keras.models.Model`
        :param kwargs: A dictionary contains other parameter settings on \
         to initialize a Keras or TensorFlow.Keras model, \
         e.g., GPU configuration.
        :type kwargs: `dict`
        """
        super().__init__(model_name, model_spec, **kwargs)

        if keras_model is None:
            if model_spec is None or (not isinstance(model_spec, dict)):
                raise ValueError('Initializing model requires '
                                 'a model specification or '
                                 'compiled keras model. '
                                 'None was provided')
            # In this case we need to recreate the model from model_spec
            self.model = self.load_model_from_spec(model_spec)
        else:
            if not issubclass(type(keras_model), (keras.models.Model,
                                                  tf.keras.models.Model)):
                raise ValueError('Compiled keras model needs to be provided '
                                 '(keras.models/tensorflow.keras.models). '
                                 'Type provided' + str(type(keras_model)))
            if not self.use_gpu_for_training or self.num_gpus == 1:
                self.model = keras_model
            elif issubclass(type(keras_model), keras.models.Model):
                from keras.utils import multi_gpu_model
                self.model = multi_gpu_model(keras_model, gpus=self.num_gpus)
                self.model.compile(optimizer=keras_model.optimizer,
                                   loss=keras_model.loss,
                                   metrics=keras_model.metrics, run_eagerly=True)
            else:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    self.model = keras_model

        # keras flag
        if issubclass(type(self.model), keras.models.Model):
            self.is_keras = True
            self.model_type = 'Keras-2.2.4'
        else:
            self.is_keras = False
            self.model_type = 'TensorFlow-Keras-1.15'
            
        # Default values for local training
        self.batch_size = 128
        self.epochs = 1
        self.steps_per_epoch = 100
        self.is_classification = True if not (model_spec and model_spec.get(
            'is_classification')) else model_spec.get('is_classification')

    def fit_model(self, train_data, fit_params=None):
        """
        Fits current model with provided training data.

        :param train_data: Training data, a tuple given in the form \
        (x_train, y_train) or a datagenerator of type `keras.utils.Sequence`, \
        `keras.preprocessing.image.ImageDataGenerator`
        :type train_data: `np.ndarray`
        :param fit_params: (optional) Dictionary with hyperparameters \
        that will be used to call Keras fit function.\
        Hyperparameter parameters should match keras expected values \
        e.g., `epochs`, which specifies the number of epochs to be run. \
        If no `epochs` or `batch_size` are provided, a default value \
        will be used (1 and 128, respectively).
        :type fit_params: `dict`
        :return: None
        """
        
        hyperparams = fit_params.get('hyperparams', {}) or {} if fit_params else {}
        local_hp = hyperparams.get('local', {}) or {}
        training_hp = local_hp.get('training', {}) or {}
        dp_hp = local_hp.get('privacy', {}) or {}
        op_hp = local_hp.get('optimizer', {}) or {}

        # Initialized with default values if not in training_hp
        batch_size = training_hp.get('batch_size', self.batch_size)
        epochs = training_hp.get('epochs', self.epochs)
        steps_per_epoch = training_hp.get('steps_per_epoch', self.steps_per_epoch)
        budget = dp_hp.get('budget', None)
        delta = dp_hp.get('delta', 0.005)
        lr = op_hp.get('lr', 0.01)
        
        logger.info('Training hps for this round => '
            'batch_size: {}, epochs {}, steps_per_epoch {}'
            .format(batch_size, epochs, steps_per_epoch))

        if epochs is None:
            logger.exception('epochs need to be provided')
            raise ModelException("Invalid hyperparams, epoch can't be None")

        try:

            if type(train_data) is tuple and type(train_data[0]) is np.ndarray:
                self.fit(
                    train_data, batch_size=batch_size, epochs=epochs, budget=budget, delta=delta, lr=lr)
            else:
                # batch_size won't be used for data generator
                self.fit_generator(train_data, epochs=epochs,
                                   steps_per_epoch=steps_per_epoch)

        except Exception as e:
            logger.exception(str(e))
            raise LocalTrainingException(
                'Error occurred while performing model.fit')

    def fit(self, train_data, batch_size, epochs, budget=None, delta=None, lr=0.01):
        """
        Fits current model using model.fit with provided training data.

        :param train_data: Training data, a tuple \
        given in the form (x_train, y_train).
        :type train_data: `np.ndarray`
        :param batch_size: Number of samples per gradient update.
        :type batch_size: `int`
        :param epochs: Number of epochs to train the model.
        :type epochs: `int`
        :return: None
        """
        x = train_data[0]
        y = train_data[1]
        if budget is None:
            self.model.fit(x, y, batch_size=batch_size, epochs=epochs)
        else:
            print('***********************************USING DP TRAINING*******************************************')
            x = np.array(x)
            y = np.array(y)
            optimizer = self.model.optimizer #(learning_rate=lr)
            loss_fn = self.model.loss
            num_batches = y.shape[0] // batch_size
            x_batches = np.array_split(x, num_batches)
            y_batches = np.array_split(y, num_batches)
            num_batches = len(y_batches)
            sigma = np.sqrt(2 * np.log(1/delta)) / budget
            for epoch in range(epochs):
                for step in range(num_batches):
                    x_batch = tf.convert_to_tensor(x_batches[step], dtype=tf.float32)
                    y_batch = tf.convert_to_tensor(y_batches[step], dtype=tf.float32)
                    with tf.GradientTape() as tape:
                        pred = self.model(x_batch)
                        loss = loss_fn(y_batch, pred)
                    grads = tape.gradient(loss, self.model.trainable_weights)
                    for g in grads:
                        length = np.prod(np.array(g.shape)) # Scale budget evenly among params
                        noise = tf.convert_to_tensor(np.ones(g.shape) * (np.random.normal(0, sigma) / length), dtype=tf.float32)
                        g += noise # Add noise to gradient
                    optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    
    def fit_generator(self, training_generator, epochs, steps_per_epoch=None):
        """
        Fits current model using model.fit_generator with provided
        training data generator.

        :param training_generator: Training datagenerator of type \
        `keras.utils.Sequence`, or \
        `keras.preprocessing.image.ImageDataGenerator`. \
        :type training_generator: `ImageDataGenerator` or \
        `keras.utils.Sequence`.
        :param epochs: Number of epochs to train the model.
        :type epochs: `int`
        :param steps_per_epoch: Total number of steps (batches of samples) \
                to yield from `generator` before declaring one epoch. \
                Optional for `Sequence` data generator` as a number of steps.
        :type steps_per_epoch: `int`
        :return: None
        """

        if type(training_generator) is NumpyArrayIterator and not steps_per_epoch:
            raise LocalTrainingException(
                "Variable steps_per_epoch cannot be None for generators not \
                    of type keras.utils.Sequence!")

        self.model.fit_generator(
            training_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

    def update_model(self, model_update):
        """
        Update keras model with provided model_update, where model_update
        should be generated according to `KerasFLModel.get_model_update()`.

        :param model_update: `ModelUpdate` object that contains the weight \
        that will be used to update the model.
        :type model_update: `ModelUpdate`
        :return: None
        """
        if isinstance(model_update, ModelUpdate):
            w = model_update.get("weights")
            self.model.set_weights(w)
        else:
            raise LocalTrainingException('Provided model_update should be of '
                                         'type ModelUpdate. '
                                         'Instead they are:' +
                                         str(type(model_update)))

    def get_model_update(self):
        """
        Generates a `ModelUpdate` object that will be sent to other entities.

        :return: ModelUpdate
        :rtype: `ModelUpdate`
        """
        w = self.model.get_weights()
        return ModelUpdate(weights=w)

    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform prediction for a batch of inputs. Note that for classification
        problems, it returns the resulting probabilities.

        :param x: Samples with shape as expected by the model.
        :type x: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param kwargs: Dictionary of keras-specific arguments.
        :type kwargs: `dict`
        :return: Array of predictions
        :rtype: `np.ndarray`
        """
        try:
            if type(x) is np.ndarray:
                preds = self.model.predict(
                    x, batch_size=batch_size, **kwargs)
            else:
                steps = self.steps_per_epoch
                if 'steps' in kwargs:
                    steps = kwargs['steps']

                if not type(x) is NumpyArrayIterator and not steps:
                    raise LocalTrainingException(
                        "Variable steps cannot be None for generator "
                        "not of type keras.utils.Sequence")
                preds = self.model.predict_generator(x, **kwargs)
        except Exception as ex:
            logger.exception(str(ex))
            raise LocalTrainingException(
                "Error occurred during prediction.")
        return preds

    def evaluate(self, test_dataset, **kwargs):
        """
        Evaluates the model given testing data.

        :param test_dataset: Testing data, a tuple given in the form \
        (x_test, y_test) or a datagenerator of type `keras.utils.Sequence`, \
        `keras.preprocessing.image.ImageDataGenerator`
        :type test_dataset: `np.ndarray`
        :param kwargs: Dictionary of metrics available for the model
        :type kwargs: `dict`
        :return: metrics
        :rtype: `dict`
        """

        if type(test_dataset) is tuple:
            x_test = test_dataset[0]
            y_test = test_dataset[1]

            return self.evaluate_model(x_test, y_test, **kwargs)

        else:
            return self.evaluate_generator_model(
                test_dataset, **kwargs)

    def evaluate_model(self, x, y, batch_size=128, **kwargs):
        """
        Evaluates the model given x and y.

        :param x: Samples with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Corresponding labels to x
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param kwargs: Dictionary of metrics available for the model
        :type kwargs: `dict`
        :return: metrics
        :rtype: `dict`
        """

        metrics = self.model.evaluate(
            x, y, batch_size=batch_size, **kwargs)
        names = self.model.metrics_names
        dict_metrics = {}
        additional_metrics = {}
        if type(metrics) == list:
            for metric, name in zip(metrics, names):
                dict_metrics[name] = metric
        else:
            dict_metrics[names[0]] = metrics

        y_pred = self.predict(x, batch_size)
        if self.is_classification:
            additional_metrics = fl_metrics.get_eval_metrics_for_classificaton(
                y, y_pred)
        else:
            additional_metrics = fl_metrics.get_eval_metrics_for_regression(
                y, y_pred)

        logger.info(additional_metrics)
        dict_metrics = {**dict_metrics, **additional_metrics}
        logger.info(dict_metrics)

        return dict_metrics

    def evaluate_generator_model(self, test_generator, **kwargs):
        """
        Evaluates the model based on the provided data generator.

        :param test_generator: Testing datagenerator of type \
        `keras.utils.Sequence`, or \
        `keras.preprocessing.image.ImageDataGenerator`.
        :type test_generator: `ImageDataGenerator` or `keras.utils.Sequence`
        :return: metrics
        :rtype: `dict`
        """

        steps = self.steps_per_epoch
        if steps in kwargs:
            steps = kwargs.get('steps')

        if not type(test_generator) is NumpyArrayIterator and not steps:
            raise LocalTrainingException(
                "Variable steps cannot be None for generator "
                "not of type keras.utils.Sequence")
            
        metrics = self.model.evaluate_generator(
            test_generator, steps=steps)
        names = self.model.metrics_names
        dict_metrics = {}
        additional_metrics = {}

        if type(metrics) == list:
            for metric, name in zip(metrics, names):
                dict_metrics[name] = metric
        else:
            dict_metrics[names[0]] = metrics

        return dict_metrics

    def save_model(self, filename=None):
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :return: filename
        :rtype: `str`
        """
        if filename is None:
            filename = 'model_{}.h5'.format(time.time())

        full_path = super().get_model_absolute_path(filename)
        self.model.save(full_path)
        logger.info('Model saved in path: %s.', full_path)
        return filename

    def load_model(self, file_name, custom_objects={}):
        """
        Loads a model from disk given the specified file_name

        :param file_name: Name of the file that contains the model to be loaded.
        :type file_name: `str`
        :param custom_objects: A dictionary of customized objects to be loaded.
        :type custom_objects: `dict`
        :return: Keras model loaded to memory
        :rtype: `keras.models.Model`
        """
        # try loading model from keras
        model = self.load_model_via_keras(file_name, custom_objects)
        if not model:
            # try loading model from tf.keras
            model = self.load_model_via_tf_keras(file_name, custom_objects)
            if model is None:
                logger.error('Loading model failed! '
                             'An acceptable compiled model should be of type '
                             '(keras.models/tensorflow.keras.models)!')
                raise FLException(
                    'Unable to load the provided compiled model!')

        return model

    def load_model_via_keras(self, file_name, custom_objects={}):
        """
        Loads a model from disk given the specified file_name via keras.

        :param file_name: Name of the file that contains the model to be loaded.
        :type file_name: `str`
        :param custom_objects: A dictionary of customized objects to be loaded.
        :type custom_objects: `dict`
        :return: Keras model loaded to memory
        :rtype: `keras.models.Model`
        """
        # try loading model from keras
        model = None
        try:
            if not self.use_gpu_for_training or self.num_gpus == 1:
                # CPU training or use only 1 GPU for training
                model = keras.models.load_model(
                    file_name, custom_objects=custom_objects)
                model._make_predict_function()
            else:
                # use multiple GPU for training
                tmp_model = keras.models.load_model(
                    file_name, custom_objects=custom_objects)
                from keras.utils import multi_gpu_model
                model = multi_gpu_model(tmp_model, gpus=self.num_gpus)
                model.compile(optimizer=tmp_model.optimizer,
                              loss=tmp_model.loss,
                              metrics=tmp_model.metrics)
        except Exception as ex:
            logger.error(
                'Loading model via keras.models.load_model failed!')

        return model

    def load_model_via_tf_keras(self, file_name, custom_objects={}):
        """
        Loads a model from disk given the specified file_name via tf.keras.

        :param file_name: Name of the file that contains the model to be loaded.
        :type file_name: `str`
        :param custom_objects: A dictionary of customized objects to be loaded.
        :type custom_objects: `dict`
        :return: tf.keras model loaded to memory
        :rtype: `tf.keras.models.Model`
        """
        # try load from tf.keras
        model = None
        try:
            if self.use_gpu_for_training:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    model = tf.keras.models.load_model(
                        file_name, custom_objects=custom_objects)
                    model._make_predict_function()
            else:
                model = tf.keras.models.load_model(
                    file_name, custom_objects=custom_objects)
                model._make_predict_function()
        except Exception as ex:
            logger.error('Loading model via tf.keras.models.load_model '
                         'failed!')

        return model

    @staticmethod
    def model_from_json_via_keras(json_file_name):
        """
        Loads a model architecture from disk via keras
        given the specified json file name.

        :param json_file_name: Name of the file that contains \
        the model architecture to be loaded.
        :type json_file_name: `str`
        :return: Keras model with only model architecture loaded to memory
        :rtype: `keras.models.Model`
        """
        # try loading model from keras
        model = None
        json_file = open(json_file_name, 'r')
        f = json_file.read()
        json_file.close()
        try:
            model = keras.models.model_from_json(f)
        except Exception as ex:
            logger.error('Loading model via '
                         'keras.models.model_from_json failed!')

        return model

    @staticmethod
    def model_from_json_via_tf_keras(json_file_name):
        """
        Loads a model architecture from disk via tf.keras
        given the specified json file name.

        :param json_file_name: Name of the file that contains \
        the model architecture to be loaded.
        :type json_file_name: `str`
        :return: tf.keras model with only model architecture loaded to memory
        :rtype: `tf.keras.models.Model`
        """
        # try loading model from keras
        model = None
        json_file = open(json_file_name, 'r')
        f = json_file.read()
        json_file.close()
        try:
            model = tf.keras.models.model_from_json(f)
        except Exception as ex:
            logger.error(
                'Loading model via tf.keras.models.model_from_json failed! ')

        return model

    def load_model_from_spec(self, model_spec):
        """
        Loads model from provided model_spec, where model_spec is a `dict`
        that contains two items: model_spec['model_architecture'] has a
        pointer to the file where the keras model architecture in stored
        in json format, and model_spec['model_weights'] contains
        the path where the associated weights are stored as h5.

        :param model_spec: A dictionary of specifications for a Keras model.
        :type model_spec: `dict`
        :return: model
        :rtype: `keras.models.Model`
        """

        if 'model_definition' in model_spec:
            model_file = model_spec['model_definition']
            model_absolute_path = config.get_absolute_path(model_file)
            custom_objects = {}
            if 'custom_objects' in model_spec:

                custom_objects_config = model_spec['custom_objects']
                for custom_object in custom_objects_config:
                    key = custom_object['key']
                    value = custom_object['value']
                    path = custom_object['path']
                    custom_objects[key] = config.get_attr_from_path(
                        path, value)

            model = self.load_model(model_absolute_path,
                                    custom_objects=custom_objects)
        else:
            # Load architecture from json file
            try:
                model = KerasFLModel.model_from_json_via_keras(
                    model_spec['model_architecture'])
                if not model:
                    model = KerasFLModel.model_from_json_via_tf_keras(
                        model_spec['model_architecture'])
                    if model is None:
                        logger.error(
                            'An acceptable compiled model should be of type '
                            '(keras.models/tensorflow.keras.models)!')
                        raise FLException(
                            'Unable to load the provided compiled model!')
            except Exception as ex:
                logger.error(str(ex))
                raise FLException(
                    'Unable to load the provided compiled model!')

            # Load weights from h5 file
            if 'model_weights' in model_spec:
                model.load_weights(model_spec['model_weights'])
            # model.load_weights(weights)

            # Compile model with provided parameters:
            compiled_option = model_spec['compile_model_options']
            try:
                if 'optimizer' in compiled_option:
                    optimizer = compiled_option['optimizer']
                else:
                    logger.warning('No optimizer information was provided '
                                   'in the compile_model_options, '
                                   'set keras optimizer to default: SGD')
                    optimizer = 'sgd'
                if 'loss' in compiled_option:
                    loss = compiled_option['loss']
                else:
                    logger.warning('No loss function was provided '
                                   'in the compile_model_options.'
                                   'set keras loss function to default: None')
                    loss = None
                if 'metrics' in compiled_option:
                    metrics = compiled_option['metrics']
                    metrics = [metrics] if isinstance(
                        metrics, str) else metrics
                else:
                    logger.warning('No metrics information was provided '
                                   'in the compile_model_options,'
                                   'set keras metrics to default: None')
                    metrics = None
                model.compile(optimizer=optimizer,
                              loss=loss,
                              metrics=metrics)
            except Exception as ex:
                logger.exception(str(ex))
                logger.exception('Failed to compiled keras model.')
        return model

    def expand_model_by_layer_name(self, new_dimension, layer_name="dense"):
        """
        Expand the current Keras model with provided dimension of
        the hidden layers or model weights.
        This method by default expands the dense layer of
        the current neural network.
        It can be extends to expand other layers specified by `layer_name`,
        for example, it can be use to increase the number of CNN filters or
        increase the hidden layer size inside LSTM.

        :param new_dimension: New number of dimensions for \
        the fully connected layers
        :type new_dimension: `list`
        :param layer_name: layer's name to be expanded
        :type layer_name: `str`
        :return: None
        """
        if new_dimension is None:
            raise FLException('No information is provided for '
                              'the new expanded model. '
                              'Please provide the new dimension of '
                              'the resulting expanded model.')

        model_config = json.loads(self.model.to_json())
        i = 0

        for layer in model_config['config']['layers']:
            # find the specified layers
            if 'class_name' in layer and \
                    layer['class_name'].strip().lower() == layer_name:
                layer['config']['units'] = new_dimension[i]
                i += 1
        if self.is_keras:
            new_model = keras.models.model_from_json(json.dumps(model_config))
        else:
            new_model = tf.keras.models.model_from_json(
                json.dumps(model_config))

        metrics = self.model.metrics_names
        if 'loss' in metrics:
            metrics.remove('loss')
        if not self.use_gpu_for_training or self.num_gpus == 1:
            new_model.compile(optimizer=self.model.optimizer,
                              loss=self.model.loss,
                              metrics=metrics)
        elif self.is_keras:
            from keras.utils import multi_gpu_model
            new_model = multi_gpu_model(new_model, gpus=self.num_gpus)
            new_model.compile(optimizer=self.model.optimizer,
                              loss=self.model.loss,
                              metrics=metrics)
        else:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                new_model.compile(optimizer=self.model.optimizer,
                                  loss=self.model.loss,
                                  metrics=metrics)

        self.model = new_model

    def is_fitted(self):
        """
        Return a boolean value indicating if the model is fitted or not.
        In particular, check if the keras model has weights.
        If it has, return True; otherwise return false.

        :return: res
        :rtype: `bool`
        """
        try:
            self.model.get_weights()
        except Exception:
            return False
        return True

    def get_weights(self):
        """
        Returns current model weights.

        :return: A list containing the current model weights.
        :rtype: `list` of `np.ndarray`
        """
        return self.model.get_weights()

    def get_loss(self, dataset):
        """
        Return the resulting loss computed based on the provided dataset.

        :param dataset: Provided dataset, a tuple given in the form \
        (x_test, y_test) or a datagenerator of type `keras.utils.Sequence`, \
        `keras.preprocessing.image.ImageDataGenerator`.
        :type dataset: `np.ndarray`
        :return: The resulting loss.
        :rtype: `float`
        """
        if 'loss' not in self.model.metrics_names:
            self.model.metrics_names.append('loss')

        res = self.evaluate(dataset)

        if 'loss' in res:
            return res['loss']
        else:
            raise FLException("Loss is not listed in the "
                              "model's metrics_names.")
