
import pandas as pd
import numpy as np
import csv
import os

os.environ["OMP_NUM_THREADS"] = '1'  


from DeepRandomCox import train_CoxKmeans
from DeepRandomCox.random_cox import predictCoxKmeans_lab
from DeepRandomCox.random_cox import predictCoxKmeans_nolab
import torch.nn.functional as F

import torch
import crypten
from support import get_omic_data
from support import sort_data
from support import mkdir
from support import plot_curve
from support import cal_pval
from torchvision import transforms
import torchvision.transforms as trans
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

is_plot = False
## is drawing
nn_config = {
    "learning_rate": 0.0001,
    "learning_rate_decay": 0.999,
    "activation": 'relu',
    "epoch_num": 5000,
    "skip_num": 1,
    "L1_reg": 1e-5,
    "L2_reg": 1e-5,
    "optimizer": 'Adam',
    "dropout": 0.0,
    "hidden_layers": [2000, 500, 24, 1],
    "standardize":True,
    "batchnorm":False,
    "momentum": 0.9,
    "n_clusters" : 2,
    "update_interval" : 1,
    "kl_rate":0,
    "ae_rate":0,
    "seed": 2
}



train_save_set = []
valid_save_set = []
test_pred_set = []
test_y_set = []
mse_set = []
ss_set = []
classify_set = []
x_reconstuct_set = []

hidden_l = [2000,1000]
lr_set = [1E-6,1E-7, 1E-8,1E-9]

dataset_lab = ['my_dataset/a/t_mrna']
dataset_nolab = ['my_dataset/b/t_mirna']
dataset_other = ['my_dataset/c/t_meth']

for filename_other in dataset_other:
    for filename_nolab in dataset_nolab:
        for filename_lab in dataset_lab:
            for h in hidden_l:
                for lr in lr_set:
                    #lab
                    valid_save_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr)])
                    mse_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr)])
                    ss_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr)])
                    classify_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr)])
                    x_reconstuct_set.append(["hidden layer : " + str(h) + " learning rate : " + str(lr)])

                    for ite_num in range(5000):
                        print("======ite_num======",ite_num)               
                        for seed in range(5000):
                            for fold_num in range(5):
                                ori_train_X_lab, ori_train_Y_lab, ori_test_X_lab, ori_test_Y_lab, ori_train_X_nolab, ori_train_Y_nolab, ori_test_X_nolab, ori_test_Y_nolab, ori_train_X_other, ori_train_Y_other, ori_test_X_other, ori_test_Y_other = get_omic_data(fea_filename_lab=(filename_lab + ".csv"), fea_filename_nolab=(filename_nolab + ".csv"),fea_filename_other=(filename_other + ".csv"), seed=seed, nfold = 4, fold_num = 0)

                                ori_idx_lab, ori_train_X_lab, ori_train_Y_lab = sort_data(ori_train_X_lab, ori_train_Y_lab)
                                ori_idx_nolab, ori_train_X_nolab, ori_train_Y_nolab = sort_data(ori_train_X_nolab, ori_train_Y_nolab)
                                ori_idx_other, ori_train_X_other, ori_train_Y_other = sort_data(ori_train_X_other, ori_train_Y_other)
                                input_nodes_lab = len(ori_train_X_lab[0])   
                                input_nodes_nolab = len(ori_train_X_nolab[0])   
                                input_nodes_other = len(ori_train_X_other[0])   
                                nn_config["learning_rate"] = lr
                                nn_config["hidden_layers"][2] = h
                                train_curve, valid_curve, model_lab, model_nolab, model_other = train_CoxKmeans(device, nn_config, input_nodes_lab, ori_train_X_lab, ori_train_Y_lab, ori_test_X_lab, ori_test_Y_lab,input_nodes_nolab, ori_train_X_nolab, ori_train_Y_nolab, ori_test_X_nolab, ori_test_Y_nolab,input_nodes_other, ori_train_X_other, ori_train_Y_other, ori_test_X_other, ori_test_Y_other)
                                valid_save_set.append(valid_curve[1])
                                mse_set.append(valid_curve[2])
                                ss_set.append(valid_curve[3])
                                classify_set.append(valid_curve[4])
                                x_reconstuct_set.append(valid_curve[5])
                                if is_plot:
                                    plot_curve(curve_data=train_curve, title="train epoch-Cindex curve")
                                    plot_curve(curve_data=valid_curve, title="test epoch-Cindex curve")

                                test_x_bar_lab, test_q_lab, prediction_lab, _ = predictCoxKmeans_lab(model_lab, device, nn_config, ori_test_X_lab)
                                test_x_bar_nolab, test_q_nolab, prediction_nolab, _ = predictCoxKmeans_nolab(model_nolab, device, nn_config, ori_test_X_nolab)
                                test_x_bar_other, test_q_other, prediction_other, _ = predictCoxKmeans_nolab(model_other, device, nn_config, ori_test_X_other)
                                prediction = (prediction_lab + prediction_nolab + prediction_other)/3.0
                                ori_test_Y = (ori_test_Y_lab + ori_test_Y_nolab + ori_test_Y_other)/3.0

                                test_pred_set.append(prediction)
                                test_y_set.append(ori_test_Y)
                        mkdir("result")
                        with open("result/" + "Cindex" + ".csv", "w",
                                newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerows(valid_save_set)
                        with open("result/" + "prediction" + ".csv", "w",
                                newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerows(test_pred_set)
                        with open("result/" + "test_Y" + ".csv", "w",
                                newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerows(test_y_set)
                        with open("result/" + "testmse" + ".csv", "w",
                                newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerows(mse_set)
                        with open("result/" + "SS" + ".csv", "w",
                                newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerows(ss_set)
                        with open("result/" + "classify" + ".csv", "w",
                                newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerows(classify_set)
                        with open("result/" + "reconsturct" + ".csv", "w",
                                newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerows(x_reconstuct_set)


							