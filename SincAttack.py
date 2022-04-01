import tensorflow as tf
from keras.layers import Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dropout,Dense, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D, BatchNormalization, Activation, Add, add
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils
import numpy as np
import math
v = K.variable(K.ones(1))
class SincConv1D(Layer):

    def __init__(
            self,
            N_filt,
            Filt_dim,
            fs,
            **kwargs):
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs

        super(SincConv1D, self).__init__(**kwargs)

    # overriding get_config method as __init__ function has positional arguements
    def get_config(self):
        return {"N_filt": self.N_filt,
                "Filt_dim": self.Filt_dim,
                "fs": self.fs}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def build(self, input_shape):
        # The filters are trainable parameters.
        self.filt_b1 = self.add_weight(
            name='filt_b1',
            shape=(self.N_filt,),
            initializer='uniform',
            trainable=True)
        self.filt_band = self.add_weight(
            name='filt_band',
            shape=(self.N_filt,),
            initializer='uniform',
            trainable=True)

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (self.fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (self.fs / 2) - 100
        self.freq_scale = self.fs * 1.0
        self.set_weights([b1 / self.freq_scale, (b2 - b1) / self.freq_scale])

        # Get beginning and end frequencies of the filters.
        min_freq = 50.0
        min_band = 50.0
        self.filt_beg_freq = K.abs(self.filt_b1) + min_freq / self.freq_scale
        self.filt_end_freq = self.filt_beg_freq + (K.abs(self.filt_band) + min_band / self.freq_scale)

        # Filter window (hamming).
        n = np.linspace(0, self.Filt_dim, self.Filt_dim)
        window = 0.54 - 0.46 * K.cos(2 * math.pi * n / self.Filt_dim)
        window = K.cast(window, "float32")

        # specifying unique name to the variable to fix issue while saving the model
        self.window = K.variable(window, name='window')


        # TODO what is this?
        t_right_linspace = np.linspace(1, (self.Filt_dim - 1) / 2, int((self.Filt_dim - 1) / 2))

        # specifying unique name to the variable to fix issue while saving the model
        self.t_right = K.variable(t_right_linspace / self.fs, name='t_right')


        super(SincConv1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):

        # filters = K.zeros(shape=(N_filt, Filt_dim))
        # Compute the filters.
        output_list = []
        for i in range(self.N_filt):
            low_pass1 = 2 * self.filt_beg_freq[i] * self.sinc(self.filt_beg_freq[i] * self.freq_scale, self.t_right)
            low_pass2 = 2 * self.filt_end_freq[i] * self.sinc(self.filt_end_freq[i] * self.freq_scale, self.t_right)
            band_pass = (low_pass2 - low_pass1)
            band_pass = band_pass / K.max(band_pass)
            output_list.append(band_pass * self.window)
        filters = K.stack(output_list)  # (80, 251)
        filters = K.transpose(filters)  # (251, 80)
        filters = K.reshape(filters, (self.Filt_dim, 1,
                                      self.N_filt))  # (251,1,80) in TF: (filter_width, in_channels, out_channels) in
        # PyTorch (out_channels, in_channels, filter_width)

        '''Given an input tensor of shape [batch, in_width, in_channels] if data_format is "NWC", or [batch, 
        in_channels, in_width] if data_format is "NCW", and a filter / kernel tensor of shape [filter_width, 
        in_channels, out_channels], this op reshapes the arguments to pass them to conv2d to perform the equivalent 
        convolution operation. Internally, this op reshapes the input tensors and invokes tf.nn.conv2d. For example, 
        if data_format does not start with "NC", a tensor of shape [batch, in_width, in_channels] is reshaped to [
        batch, 1, in_width, in_channels], and the filter is reshaped to [1, filter_width, in_channels, out_channels]. 
        The result is then reshaped back to [batch, out_width, out_channels] (where out_width is a function of the 
        stride and padding as in conv2d) and returned to the caller. '''

        # Do the convolution.

        out = K.conv1d(
            x,
            kernel=filters
        )

        return out

    def sinc(self,band, t_right):
        y_right = K.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
        # y_left = flip(y_right, 0) TODO remove if useless
        y_left = K.reverse(y_right, 0)
        #K.variable(K.ones(1))
        y = K.concatenate([y_left, v, y_right])
        return y

    def compute_output_shape(self, input_shape):
        new_size = conv_utils.conv_output_length(
            input_shape[1],
            self.Filt_dim,
            padding="same",
            stride=1,
            dilation=1)
        return (input_shape[0],) + (new_size,) + (self.N_filt,)

"""
占用大量CPU内存，准备时间非常长，可能造成内存溢出。只能使用小批。
"""
from tensorflow.keras.models import Model, load_model
from aisca_v2.core import KerasNNPredictModel, StandardAttackModel
from aisca_v2.core.hyper_para import HyperPara
from aisca_v2.core.predictmodels import _predict_key_scores


class SincNet(KerasNNPredictModel):
    def __init__(self, parent_attack_model, name, for_encryption, intermediates, post_processors):
        super().__init__(parent_attack_model, name, for_encryption, intermediates, post_processors)
        self.pca_components = (0)  # 0: 不做PCA处理
        self.hyper_paras.add(self, 'pca_components', '主成分的数量', HyperPara.INT, [0, 100],
                             comment='主成分分析可替代使用兴趣点降维。0表示不采用PCA处理。'
                                     '当使用兴趣点时，不进行PCA处理，本参数不起作用。',
                             discretization=('even', 10))
        self.batch_size = 2048
        self.hyper_paras.add(self, 'batch_size', '训练批大小', HyperPara.INT_ENUM,
                             (512, 1024, 2048, 4096, 5120, 6144, 7168, 8192),
                             comment='对信噪比很低的数据，使用尽可能大的训练批（如4096, 5120等）。'
                                     '但批大小越大，占用内存越多。使用大的训练批时，应根据设备的CPU或GPU内存，适当降低训练的并行度。')

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
        input_size = data_generator.batch_shape[1]
        input = Input(shape=data_generator.batch_shape[1])
        input = Reshape([input_size, 1])(input)
        model_name = "stack_sincnn_"
        x = SincConv1D(1, 11, 16000, name=model_name+"sinc")(input)
        # Block 1
        x = Conv1D(64, 11, activation='relu', padding='same', name=model_name+'block1_conv1')(x)
        x = AveragePooling1D(2, strides=2, name=model_name+'block1_pool')(x)
        # Block 2
        x = Conv1D(128, 11, activation='relu', padding='same', name=model_name+'block2_conv1')(x)
        x = AveragePooling1D(2, strides=2, name=model_name+'block2_pool')(x)
        # Block 3
        x = Conv1D(256, 11, activation='relu', padding='same', name=model_name+'block3_conv1')(x)
        x = AveragePooling1D(2, strides=2, name=model_name+'block3_pool')(x)
        # Block 4
        x = Conv1D(512, 11, activation='relu', padding='same', name=model_name+'block4_conv1')(x)
        x = AveragePooling1D(2, strides=2, name=model_name+'block4_pool')(x)
        # Block 5
        x = Conv1D(512, 11, activation='relu', padding='same', name=model_name+'block5_conv1')(x)
        x = AveragePooling1D(2, strides=2, name=model_name+'block5_pool')(x)
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name=model_name+'fc1')(x)
        x = Dropout(0.4)(x)
        x = Dense(4096, activation='relu', name=model_name+'fc2')(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='softmax', name=model_name+'predictions')(x)

        # Create model.
        model = Model(input, x, name='sinc_best')
        optimizer = RMSprop(lr=0.00001)
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer=optimizer, metrics='accuracy')
        return model


class SincAttackModel(StandardAttackModel):
    def __init__(self, owner_attack_instance, rnd, sbyte, for_encryption, predict_model_intermediates,
                 predict_model_post_processors):
        super().__init__(owner_attack_instance, rnd, sbyte, for_encryption, predict_model_intermediates,
                         predict_model_post_processors)

    @classmethod
    def get_title(cls):
        return 'SINC_CNN'

    @classmethod
    def get_predict_model_class(cls):
        return SincNet

    def predict_key_scores(self, attack_datafile, attack_size, n_attacks, from_trace=0, real_key=None, **kwargs):
        return _predict_key_scores(self.predict_model, attack_datafile, attack_size, n_attacks, from_trace)
