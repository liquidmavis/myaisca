from aisca_v2.core import KerasNNPredictModel, HyperPara
from aisca_v2.core.attackmodels import StandardAttackModel
from aisca_v2.core.predictmodels import _predict_key_scores
class CNN1DAttackModel(StandardAttackModel):
    def __init__(self, owner_attack_instance, rnd, sbyte, for_encryption, predict_model_intermediates,
                 predict_model_post_processors):
        super().__init__(owner_attack_instance, rnd, sbyte, for_encryption, predict_model_intermediates,
                         predict_model_post_processors)

    @classmethod
    def get_title(cls):
        return "stack一维卷积"

    @classmethod
    def get_predict_model_class(cls):
        return CNN1D_STACK

    def predict_key_scores(self, attack_datafile, attack_size, n_attacks, from_trace=0, real_key=None, **kwargs):
        return _predict_key_scores(self.predict_model, attack_datafile, attack_size, n_attacks, from_trace)

from aisca_v2.lib.neuralnetwork import NetworkDefine
from aisca_v2.lib import get_logger
logger = get_logger()


class CNN1D_STACK(KerasNNPredictModel):
    def __init__(self, parent_attack_model, name, for_encryption, intermediates, post_processors):
        super().__init__(parent_attack_model, name, for_encryption, intermediates, post_processors)

        # # 替换缺省的PCA和Normalization设置
        # self.pca_components = 0  # 0: 不做PCA处理
        # self.hyper_paras.add(self, 'pca_components', '主成分的数量', HyperPara.INT, (0, 0),
        #                      comment='CNN1D不支持进行PCA处理。',
        #                      discretization=None)
        self.hyper_paras.update_pca_components(valid_value=(0, 150), default_value=(20, 120), value=0)

        self.convolution_layers = 0  # 0 represent auto generate necessary blocks
        self.hyper_paras.add(self, 'convolution_layers', '卷积层数', HyperPara.INT, [0, 10],
                             comment='0表示自动计算卷积层数。')

        self.init_filters = 4
        self.hyper_paras.add(self, 'init_filters', '第一卷积层输出特征数量', HyperPara.INT, [2, 16],
                             comment='第一卷积层的输出特征是能迹的原始特征。该值过小，会遗漏原始特征，造成模型欠拟合。'
                                     '该值越大，卷积模型占用内存越多，训练越慢。',
                             discretization=('times', 2))
        self.max_filters = (64, 256)
        self.hyper_paras.add(self, 'max_filters', '最后卷积层输出特征数量', HyperPara.INT, [64, 256],
                             comment='信噪比越高，特征越多（如128, 256）；信噪比越低，特征越少（如32, 64）',
                             discretization=('times', 2))

        self.activation = 'selu'
        self.hyper_paras.add(self, 'activation', '激活函数', HyperPara.FUNCTION_ENUM, ['selu', 'relu'],
                             comment='selu、relu是比较卷积模型中常见的选择。selu更稳定。')

        self.first_kernel_size = (5, 15)
        self.hyper_paras.add(self, 'first_kernel_size', '第一卷积层卷积核宽度', HyperPara.INT, [3, 21],
                             comment='能迹抖动越大，第一卷积成的卷积核应该越大(以便分辨错位情况下各卷积区域能耗组合的特征)。'
                                     '无抖动时，设置为7较好。',
                             discretization=('even', 2), default_value=7)
        self.first_pool_size = 2
        self.hyper_paras.add(self, 'first_pool_size', '初始卷积块的池化大小', HyperPara.INT, [2, 4],
                             comment='池化大小一般取2。池化操作以池化大小减小卷积层的宽度。'
                                     '池化大小越大，可能的卷积层数越少。而卷积模型一般需要较多的卷积层数量。',
                             discretization=('times', 2))
        self.pooling_function = ('average_pooling1d', 'max_pooling1d')
        self.hyper_paras.add(self, 'pooling_function', '池化方法', HyperPara.STR_ENUM,
                             ['average_pooling1d', 'max_pooling1d'],
                             comment='池化层是卷积模型容忍能迹样本不对齐的关键。'
                                     '在能迹不对齐的情况下，一般最大池化优于平均池化。')
        self.dense_layers = 1
        self.hyper_paras.add(self, 'dense_layers', '全连接层数', HyperPara.INT, (0, 2),
                             comment='一般卷积结构中包含一层全连接层。但有时也可以不要全连接层。',
                             discretization=('even', 1))
        self.dense_layer_size = (64, 128)
        self.hyper_paras.add(self, 'dense_layer_size', '全连接层大小', HyperPara.INT, (64, 256),
                             comment='全连接层大小可以等于卷积的输出特征数，'
                                     '或介于卷积输出特征与分类数（攻击目标的取值数量）之间。',
                             discretization=('even', 16))

    def get_target_name(self):
        return self._intermediates[0]

    def get_input_names(self):
        return ['traces']

    # def verify_suggestion(self, suggestion, dataset):
    #     # 步长 < 核大小
    #     # kernel_size = HyperPara.get_hyper_para(suggestion, 'first_kernel_size').value
    #     # stride = HyperPara.get_hyper_para(suggestion, 'first_block_stride').value
    #     # if stride > kernel_size:
    #     #     return False
    #     # kernel_size = HyperPara.get_hyper_para(suggestion, 'rest_kernel_size').value
    #     # stride = HyperPara.get_hyper_para(suggestion, 'rest_block_stride').value
    #     # if stride > kernel_size:
    #     #     return False
    #
    #     # 保证卷积输出的神经元数量在合理的范围内（层大小 x _ext），如: >=64, <=1024
    #     # 剩余块的卷积核大小 > 第一层卷积核大小
    #     # 容量不变？
    #     # 卷积核大小<=层大小
    #     return super(CNN1D, self).verify_suggestion(suggestion, dataset)

    def prepare_target(self, dataset, for_training):
        self.set_target_name(self.intermediates[0])
        if for_training:
            self.update_target_classes(dataset)

    @classmethod
    def multi_intermediates(cls):
        return False

    @classmethod
    def intermediate_selectable(cls):
        return True

    @classmethod
    def compatible_with_trace(cls, trace_shape):
        return len(trace_shape) == 1

    @staticmethod
    def _get_layer_size(input_size, strides, padding):
        n_steps = input_size // strides
        rest_size = input_size % strides
        if padding == 'same' and rest_size > 0:
            n_steps += 1
        return n_steps

    # TODO: _define_conv1d_block没有使用，应该移到SubNetworkDefine中
    @staticmethod
    def _define_conv1d_block(sub_network, kernel_size, conv_stride, filters, activation,
                             pooling_function, pool_size, pool_stride, padding,
                             block_idx, layer_size, prv_layer=None):
        block_def = list()
        conv_layer = sub_network.add_conv1d_layer(f'conv{block_idx}', kernel_size, conv_stride, filters,
                                                  activation, padding)
        if prv_layer is not None:
            conv_layer['previous'] = prv_layer
        block_def.append(conv_layer)

        layer_size[0] = CNN1D_STACK._get_layer_size(layer_size[0], conv_stride, padding)
        logger.debug(f'Conv{block_idx}: {layer_size[0]}')

        if pool_size > 0 and pooling_function is not None:
            pool_layer = sub_network.add_pooling_layer(f'pool{block_idx}', pooling_function,
                                                       pool_size, pool_stride, padding)
            block_def.append(pool_layer)
            layer_size[0] = CNN1D_STACK._get_layer_size(layer_size[0], pool_stride, padding)
            logger.debug(f'Pool{block_idx}: {layer_size[0]}')
        return block_def



    def define_keras_model(self, data_generator):
        import tensorflow as tf
        import numpy as np

        input_size = data_generator.batch_shape[1]
        target = self.get_target_name()
        classes = self.num_classes

        # calculate necessary blocks
        n_blocks = 0
        # init_filters = self.max_filters
        # calculate the number of Conv blocks to shrink layer_size to 1.
        layer_size = input_size
        while layer_size > 1:
            layer_size = int(np.ceil(layer_size / 2))
            n_blocks += 1
            # init_filters = init_filters // 2
        # init_filters = max(self.init_filters, self.max_filters // (2 ** (n_blocks - 1)))
        pure_conv_blocks = 0
        # init_filters = 8  # fix to 8, ensure first pure conv layers are the same
        # convert to pure convolution blocks
        # first_filters = init_filters
        if n_blocks < self.convolution_layers:
            # 需要添加一些纯卷积层（不做池化处理）
            n_blocks = self.convolution_layers
            pure_conv_blocks = self.convolution_layers - n_blocks  # blocks that don't need pooling to shrink size
            # first_filters = max(2, self.max_filters // (2 ** (n_blocks - 1)))
        elif n_blocks > self.convolution_layers > 0:
            n_blocks = self.convolution_layers
            # first_filters = init_filters
            pure_conv_blocks = 0
        model_name = "stack_cnn"
        input_layer = tf.keras.layers.Input(input_size, name=model_name+"_input")
        layer = tf.keras.layers.Reshape([input_size, 1])(input_layer)

        layer_size = input_size
        # i = 0
        # filters = first_filters
        filters = self.init_filters
        # dropout = self.dropout
        padding = 'same'
        pool_size = self.first_pool_size
        kernel_size = self.first_kernel_size
        # while layer_size > 1:  # loop until layer size becomes 1
        for i in range(n_blocks):
            kernel_size = min(layer_size, kernel_size)
            pool_size = min(layer_size, pool_size)

            layer = tf.keras.layers.Conv1D(
                filters,
                kernel_size,
                strides=1,
                padding=padding,
                activation=self.activation,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale),
                name=model_name+f'_Conv1D_{i}'
            )(layer)
            if self.batch_normalization:
                layer = tf.keras.layers.BatchNormalization()(layer)
            # update input_size
            # layer_size = layer.shape[1]
            if pure_conv_blocks == 0:  # indicate those layers that use pooling to shrink
                if self.pooling_function == 'average_pooling1d':
                    layer = tf.keras.layers.AveragePooling1D(pool_size, strides=2, padding=padding, name=model_name+f'_poll{i}')(layer)
                else:
                    layer = tf.keras.layers.MaxPooling1D(pool_size, strides=2, padding=padding, name=model_name+f'_pool{i}')(layer)
                # layer_size = layer.shape[1]
                # 特征数倍增
                filters *= 2
                if filters > self.max_filters:
                    filters = self.max_filters
                # 卷积核按3减小，池化大小按2减小。
                kernel_size = max(3, kernel_size - 3)
                pool_size = max(2, pool_size - 2)
            else:
                # 纯卷积层的特征数保持在初始特征数以下
                # # 特征数倍增
                # filters *= 2
                # if filters > init_filters:
                #     filters = init_filters
                kernel_size = max(3, kernel_size - 3)
                pure_conv_blocks -= 1
            i += 1
        layer = tf.keras.layers.Flatten()(layer)
        # dense layers
        for i in range(self.dense_layers):
            layer = tf.keras.layers.Dropout(self.dropout,name=model_name+"_dropou_0")(layer)
            layer = tf.keras.layers.Dense(
                self.dense_layer_size,
                kernel_regularizer=tf.keras.regularizers.L2(self.l2_scale),
                name=model_name+f'_dense{i}'
            )(layer)
            if self.batch_normalization:
                layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.keras.layers.Activation(self.activation,name=model_name+f'_activation_{i}')(layer)

        layer = tf.keras.layers.Dropout(self.dropout,name=model_name+"_dropout_1")(layer)
        layer = tf.keras.layers.Dense(classes,
                                       kernel_regularizer=tf.keras.regularizers.l2(self.l2_scale),
                                       name=model_name+"_dense_last")(layer)
        if self.batch_normalization:
            layer = tf.keras.layers.BatchNormalization()(layer)
        output = tf.keras.layers.Activation('softmax',name=model_name+"_activation_last")(layer)
        model = tf.keras.Model(input_layer, output, name='STACK_CNN1D')
        model.compile('adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics='accuracy')
        model.summary()
        return model
