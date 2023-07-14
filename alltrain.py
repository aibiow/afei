import pandas as pd
import numpy as np
import csv

from DeepRandomCox import train_CoxKmeans
from DeepRandomCox import predictCoxKmeans
from DeepRandomCox import getCindex


import torch
from support import load_omic_data
from support import sort_data
from support import mkdir
from support import plot_curve
from support import cal_pval
from idle_gpu import idle_gpu
### 指定使用的GPU

# gpu_id = idle_gpu()
device = torch.device("cuda:{}".format(idle_gpu()) if torch.cuda.is_available() else "cpu")
print(device)

##是否保存模型
is_save = True
##是否加载模型
is_load= True


### 是否画图
is_plot = False
## is drawing
nn_config = {
    "learning_rate": 0.0000007, #0.0000007
    "learning_rate_decay": 0.999,
    "activation": 'relu',
    "epoch_num": 1000,
    "skip_num": 5,
    "L1_reg": 1e-5,
    "L2_reg": 1e-5,
    "optimizer": 'Adam',
    "dropout": 0.0,
    "hidden_layers": [500, 200, 24, 1],
    "standardize":False,
    "batchnorm":False,
    "momentum": 0.9,
    "n_clusters" : 2,
    "update_interval" : 1,
    "kl_rate":0,
    "ae_rate":0,
    "seed": 1
}



# if is_plot:
#     plot_curve(train_curve, title="train epoch-Cindex curve", x_label="epoch", y_label="Cindex")
#     plot_curve(test_curve, title="test epoch-Cindex curve", x_label="epoch", y_label="Cindex")

dataset = ["my_dataset/test"]

# ### 5-independent TEST
train_save_set = []
train_pred_set = []

mse_set = []
ss_set = []
classify_set = []
x_reconstuct_set = []

hidden_l = [10]
# lr_set = [1E-6]
lr_set = [1E-6]
# hidden_l = [16,20,24,28]
# hidden_l = [100, 50, 20, 10, 5]
# lr_set = [0.0000001,0.0000005,0.000001]

if is_save:
    for filename in dataset:
        for h in hidden_l:
            for lr in lr_set:
                train_save_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
                mse_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
                ss_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
                classify_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])
                x_reconstuct_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr) + filename])

                ori_train_X, ori_train_Y, headers = load_omic_data(filename + ".csv")
                ori_idx, ori_train_X, ori_train_Y = sort_data(ori_train_X, ori_train_Y)
                input_nodes = len(ori_train_X[0])
                nn_config["learning_rate"] = lr
                nn_config["hidden_layers"][2] = h

                blank, train_curve, model = train_CoxKmeans(device, nn_config, input_nodes, ori_train_X, ori_train_Y, ori_train_X, ori_train_Y)
                train_save_set.append(train_curve[1])
                mse_set.append(train_curve[2])
                ss_set.append(train_curve[3])
                classify_set.append(train_curve[4])
                x_reconstuct_set.append(train_curve[5])


                train_x_bar, train_q, prediction = predictCoxKmeans(model, device, nn_config, ori_train_X)
                train_pred_set.append(prediction)

                # 保存
                mkdir("model_save")
                torch.save(model, 'model_save/model.pkl')




                mkdir("result_all")
                with open("result_all/" + "Cindex" + ".csv", "w",
                          newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(train_save_set)
                with open("result_all/" + "prediction" + ".csv", "w",
                          newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(train_pred_set)
                with open("result_all/" + "trainmse" + ".csv", "w",
                          newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(mse_set)
                with open("result_all/" + "SS" + ".csv", "w",
                          newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(ss_set)
                with open("result_all/" + "classify" + ".csv", "w",
                          newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(classify_set)
                with open("result_all/" + "reconsturct" + ".csv", "w",
                          newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(x_reconstuct_set)

    ### write csv
###

if is_load:
    indenpendent_pred_set = []
    indenpendent_cindex_set = []
    indenpendent_cindex = []
    filename = "my_dataset/ov_all"
    model = torch.load('model_save/model.pkl')
    ori_train_X, ori_train_Y, headers = load_omic_data(filename + ".csv")
    train_x_bar, train_q, prediction = predictCoxKmeans(model, device, nn_config, ori_train_X)
    prediction = prediction.reshape(1,-1)
    indenpendent_pred_set.append(prediction)
    indenpendent_cindex.append(getCindex(ori_train_Y, prediction))
    indenpendent_cindex_set.append(indenpendent_cindex)
    print(indenpendent_cindex)
    with open("result_all/" + "indenpendent_prediction" + ".csv", "w",
              newline='') as csvfile:
        writer = csv.writer(csvfile)
        for pred in prediction:
            writer.writerow(pred)
    with open("result_all/" + "indenpendent_Cindex" + ".csv", "w",
              newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(indenpendent_cindex_set)
### 参数
### 5-cv
### time-Death graph-done
### censor-experiment
### return a loss function Cindex graph-done
### p-value
