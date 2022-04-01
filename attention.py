"""
占用大量CPU内存，准备时间非常长，可能造成内存溢出。只能使用小批。
"""

from aisca_v2.core import KerasNNPredictModel, StandardAttackModel
from aisca_v2.core.hyper_para import HyperPara
from aisca_v2.core.predictmodels import _predict_key_scores


class Attention(KerasNNPredictModel):
    def __init__(self, parent_attack_model, name, for_encryption, intermediates, post_processors):
        super().__init__(parent_attack_model, name, for_encryption, intermediates, post_processors)

        self.hyper_paras.update_pca_components(valid_value=(0, 150), default_value=(20, 120), value=0)
        self.hyper_paras['batch_size'].update(valid_value=(16, 32, 64, 256, 512, 1024, 2048, 4096, 5120, 6144, 7168, 8192),
                                              default_value=32, value=32)

        self.state_dimension = 4
        self.hyper_paras.add(self, 'state_dimension', 'LSTM状态维度', HyperPara.INT, [4, 256],
                             comment='', discretization=('even', 1))

        self.lstm_activation = 'tanh'
        self.recurrent_activation = 'sigmoid'
        # self.seq_length = 256  # 序列长度（是否应该是整个能迹的长度？）

        self.dense_layers = 1
        self.hyper_paras.add(self, 'dense_layers', '全连接层数', HyperPara.INT, (0, 2),
                             comment='一般卷积结构中包含一层全连接层。但有时也可以不要全连接层。',
                             discretization=('even', 1))
        self.dense_layer_size = (64, 128)
        self.dense_activation = 'tanh'
        self.hyper_paras.add(self, 'dense_layer_size', '全连接层大小', HyperPara.INT, (64, 256),
                             comment='全连接层大小可以等于卷积的输出特征数，'
                                     '或介于卷积输出特征与分类数（攻击目标的取值数量）之间。',
                             discretization=('even', 16))

    @classmethod
    def multi_intermediates(cls):
        return False

    @classmethod
    def intermediate_selectable(cls):
        return True

    @classmethod
    def compatible_with_trace(cls, trace_shape):
        return len(trace_shape) == 1

    def verify_suggestion(self, suggestion, dataset):
        return True

    def prepare_target(self, dataset, for_training):
        self.set_target_name(self.intermediates[0])
        if for_training:
            self.update_target_classes(dataset)

    def prepare_inputs(self, dataset, for_training):
        self.set_input_names('traces')

    def define_keras_model(self, data_generator):
        from tensorflow.keras import layers, Model, regularizers, losses

        trace_size = data_generator.batch_shape[1]
        target = self.get_target_name()
        classes = self.num_classes

        inputs = layers.Input(trace_size, name=self.get_input_names()[0])
        # input shape of LSTM should be (batch, timestamp, dimension=1)
        layer = layers.Reshape((trace_size, 1))(inputs)
        # layer = layers.LSTM(self.state_dimension)(layer)
        layer = layers.LSTM(self.state_dimension, return_sequences=True, return_state=False)(layer)

        # encoding to shorter sequence by CNN1D
        # layer = layers.Conv1D(4, kernel_size=5, activation='relu')(layer)
        # layer = layers.MaxPooling1D(2, strides=2)(layer)
        # layer = layers.Conv1D(8, kernel_size=3, activation='relu')(layer)
        # layer = layers.MaxPooling1D(2, strides=2)(layer)   # 1/4 length
        # self-attention
        layer = layers.MultiHeadAttention(1, key_dim=self.state_dimension//2, dropout=0.5)(
            layer, layer, return_attention_scores=False)
        # print(layer.shape)
        layer = layers.Flatten()(layer)
        # dense layers
        for i in range(self.dense_layers):
            layer = layers.Dropout(self.dropout)(layer)
            layer = layers.Dense(
                self.dense_layer_size,
                kernel_regularizer=regularizers.L2(self.l2_scale),
                name=f'dense{i}'
            )(layer)
            layer = layers.BatchNormalization()(layer)
            layer = layers.Activation(self.dense_activation)(layer)

        layer = layers.Dropout(self.dropout)(layer)
        layer = layers.Dense(classes,
                             kernel_regularizer=regularizers.l2(self.l2_scale),
                             name=target)(layer)
        if self.batch_normalization:
            layer = layers.BatchNormalization()(layer)
        outputs = layers.Activation('softmax')(layer)
        model = Model(inputs, outputs, name='SelfAttention')
        model.compile('adam',
                      loss=losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics='accuracy')
        model.summary()
        return model


class AttentionAttackModel(StandardAttackModel):
    def __init__(self, owner_attack_instance, rnd, sbyte, for_encryption, predict_model_intermediates,
                 predict_model_post_processors):
        super().__init__(owner_attack_instance, rnd, sbyte, for_encryption, predict_model_intermediates,
                         predict_model_post_processors)

    @classmethod
    def get_title(cls):
        return 'Self-Attention'

    @classmethod
    def get_predict_model_class(cls):
        return Attention

    def predict_key_scores(self, attack_datafile, attack_size, n_attacks, from_trace=0, real_key=None, **kwargs):
        return _predict_key_scores(self.predict_model, attack_datafile, attack_size, n_attacks, from_trace)
