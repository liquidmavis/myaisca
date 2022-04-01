import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers, losses
from tensorflow.keras.applications.xception import Xception

def define_keras_model():
    """
    如何定义多输出的keras模型的例子：用于定义基础模型
    包括：特征层、logits层和概率层
    :return:
    """
    inputs = layers.Input(100)
    # 两个隐层，最后一个作为第一个输出
    x = layers.Dense(32, 'selu')(inputs)
    features = layers.Dense(32, 'selu')(x)
    logits = layers.Dense(256)(features)
    probs = layers.Activation('softmax')(logits)

    # 定义有3个输出的模型
    model = Model(inputs=inputs, outputs=(features, logits, probs))

    model.compile('adam', loss=losses.SparseCategoricalCrossentropy(), metric='accuracy')
    return model


def define_keras_model_2():
    """
    加载预训练模型，定义综合模型
    :return:
    """
    model_files = ['', '', '']
    pretrained_models = []
    for model_file in model_files:
        pretrained_models.append(load_model(model_file))

    # 获得各模型的输入层
    inputs = []
    for model in pretrained_models:
        inputs.append(model.input)

    # 根据实验需要，取各模型的不同输出层
    # 假设使用特征层作为综合模型的输入，即模型output列表中的0号元素。
    outputs = []
    for model in pretrained_models:
        outputs.append(layers.Flatten()(model.output[0]))
    # 把预训练的输出拼接在一起
    pretrained_output = layers.Concatenate()(outputs)

    # 定义综合模型自身的结构（假设采用全连接）
    x = layers.Dense(128, 'selu')(pretrained_output)
    probs = layers.Dense(256)(x, 'softmax')

    # 综合模型使用所有基础模型的输入（多个）作为输入，输出probs。
    # TODO：效率问题。如果多个输入模型的输入是相同的，那么需要重复提供这些输入。
    model = Model(inputs, probs)

    model.compile('adam', loss=losses.SparseCategoricalCrossentropy(), metric='accuracy')

    return model


def define_keras_model_3():
    """
    加载预训练模型，定义综合模型
    优化输入：假设各基础模型的输入是一样的
    原理：keras模型可以被当作一个层来被调用
    :return:
    """
    model_files = ['', '', '']
    pretrained_models = []
    for model_file in model_files:
        pretrained_models.append(load_model(model_file))

    # 假设各模型的输入层形状是一样的
    input_shape = pretrained_models[0].input.shape

    # 定义新的输入层
    input = layers.Input(input_shape)

    # 调用各模型
    outputs = []
    for model in pretrained_models:
        # 根据实验需要，取各模型的不同输出层
        # 假设使用特征层作为综合模型的输入，即模型output列表中的0号元素。
        _model_output = model(input)[0]
        _model_output = layers.Flatten()(_model_output)
        outputs.append(_model_output)
    # 把预训练的输出拼接在一起
    pretrained_output = layers.Concatenate()(outputs)

    # 定义综合模型自身的结构（假设采用全连接）
    x = layers.Dense(128, 'selu')(pretrained_output)
    probs = layers.Dense(256)(x, 'softmax')

    # 使用新定义的输入层作为模型输入，输出probs。
    model = Model(input, probs)

    model.compile('adam', loss=losses.SparseCategoricalCrossentropy(), metric='accuracy')

    return model