"""
占用大量CPU内存，准备时间非常长，可能造成内存溢出。只能使用小批。
"""
from tensorflow.keras.models import Model, load_model
from aisca_v2.core import KerasNNPredictModel, StandardAttackModel
from aisca_v2.core.hyper_para import HyperPara
from aisca_v2.core.predictmodels import _predict_key_scores


class STACKING(KerasNNPredictModel):
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
        from tensorflow.keras import layers, Model, regularizers, losses
        """
            加载预训练模型，定义综合模型
            优化输入：假设各基础模型的输入是一样的
            原理：keras模型可以被当作一个层来被调用
            :return:
            """
        root_path = "/home/liuyang/aisca-v2/liuyang-demo/MyStack"
        model_file = root_path+'/CNN1DAttackModel-CNN1D(SBoxOut)/rnd0_sbyte0-model/CNN1D/job-75d4bd97/model.h5'


        pre_model = load_model(model_file)

        # 假设各模型的输入层形状是一样的
        input_shape = pre_model.input.shape

        # 定义新的输入层
        input = layers.Input(shape=(input_shape[1],),name="stack_input")

        _model_output = pre_model(input)
        # 取（None,256）的256作为拼接
        _model_output = layers.Flatten(name="stack_flatten")(_model_output)
        # 定义综合模型自身的结构（假设采用全连接）
        x = layers.Dense(128, 'selu',name="stack_dense0")(_model_output)
        probs = layers.Dense(256,name="stack_dense1")(x)

        # 使用新定义的输入层作为模型输入，输出probs。
        model = Model(input, probs)

        model.compile('adam', loss=losses.SparseCategoricalCrossentropy(), metrics='accuracy')

        return model


class StackAttackModel(StandardAttackModel):
    def __init__(self, owner_attack_instance, rnd, sbyte, for_encryption, predict_model_intermediates,
                 predict_model_post_processors):
        super().__init__(owner_attack_instance, rnd, sbyte, for_encryption, predict_model_intermediates,
                         predict_model_post_processors)

    @classmethod
    def get_title(cls):
        return '测试stacking'

    @classmethod
    def get_predict_model_class(cls):
        return STACKING

    def predict_key_scores(self, attack_datafile, attack_size, n_attacks, from_trace=0, real_key=None, **kwargs):
        return _predict_key_scores(self.predict_model, attack_datafile, attack_size, n_attacks, from_trace)
