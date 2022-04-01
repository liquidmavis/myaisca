from tensorflow.keras.models import Model, load_model
root_path = "/home/liuyang/aisca-v2/liuyang-demo/MyStack"
model_files = [root_path+'/CNN1DAttackModel-CNN1D(SBoxOut)-1/rnd0_sbyte2-model/CNN1D/job-cc51a728/model.h5'
                       ,root_path+'/MLPAttackModel-MLP(SBoxOut)/rnd0_sbyte0-model/MLP/job-12d9f936/model.h5']
for model_file in model_files:
    model = load_model(model_file)
    model.summary()
