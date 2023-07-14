import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
from lifelines.utils import concordance_index as ci
from lifelines.statistics import logrank_test
import torch
#import random

def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + 'Folder create successfully !')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' Folader is exist')
        return False

def cal_pval(time, pred):
    event = np.zeros_like(time)
    event[time > 0] = 1
    pred_median = np.median(pred)
    risk_group = np.zeros_like(pred)
    risk_group[pred > pred_median] = 1

    group_lowrisk_time = time[risk_group==0].copy()
    group_highrisk_time = time[risk_group==1].copy()
    group_lowrisk_event = event[risk_group==0].copy()
    group_highrisk_event = event[risk_group==1].copy()

    results = logrank_test(group_lowrisk_time, group_highrisk_time, event_observed_A=group_lowrisk_event , event_observed_B=group_highrisk_event)
    # results.print_summary()
    return results.p_value



def sort_data(X,Y):
    T = - np.abs(np.squeeze(np.array(Y)))  #Y是标签
    sorted_idx = np.argsort(T)     #np.argsort(a) 将a中的元素从小到大排列，提取其在排列前对应的index(索引)输出

    return sorted_idx, X[sorted_idx], Y[sorted_idx]


# def get_omic_data_lab(fea_filename, seed, nfold = 5, fold_num =0):
#     X_lab, Y_lab, H_lab = load_omic_data_lab(fea_filename)   #X=data_fea, Y=data_time, H=headers
#     train_X_lab, train_Y_lab, test_X_lab, test_Y_lab = split_data(spilt_seed=seed, fea=X_lab, label=Y_lab, nfold=nfold, fold_num=fold_num)
#     return train_X_lab, train_Y_lab, test_X_lab, test_Y_lab

def get_omic_data(fea_filename_lab, fea_filename_nolab, fea_filename_other, seed, nfold, fold_num = 0):
    X_lab, Y_lab, H_lab = load_omic_data_lab(fea_filename_lab)   #X=data_fea, Y=data_time, H=headers
    X_nolab, Y_nolab, H_nolab = load_omic_data_nolab(fea_filename_nolab)   #X=data_fea, Y=data_time, H=headers
    X_other, Y_other, H_other = load_omic_data_other(fea_filename_other)   #X=data_fea, Y=data_time, H=headers
    train_X_lab, train_Y_lab, test_X_lab, test_Y_lab, train_X_nolab, train_Y_nolab, test_X_nolab, test_Y_nolab, train_X_other, train_Y_other, test_X_other, test_Y_other = split_data_sum(spilt_seed=seed, fea_lab=X_lab, label_lab=Y_lab, fea_nolab=X_nolab, label_nolab=Y_nolab, fea_other=X_other, label_other=Y_other, nfold = nfold, fold_num = fold_num)
    #train_X, train_Y, test_X, test_Y = split_data(spilt_seed=seed, fea=X_lab, label=Y_lab, nfold=nfold, fold_num=fold_num)
    #train_X_nolab, train_Y_nolab, test_X_nolab, test_Y_nolab = split_data(spilt_seed=seed, fea=X_nolab, label=Y_nolab, nfold=nfold, fold_num=fold_num)
    return train_X_lab, train_Y_lab, test_X_lab, test_Y_lab, train_X_nolab, train_Y_nolab, test_X_nolab, test_Y_nolab, train_X_other, train_Y_other, test_X_other, test_Y_other

# def get_omic_data_nolab(fea_filename, seed, nfold = 5, fold_num =0):
#     X_nolab, Y_nolab, H_nolab = load_omic_data_nolab(fea_filename)   #X=data_fea, Y=data_time, H=headers
#     train_X_nolab, train_Y_nolab, test_X_nolab, test_Y_nolab = split_data(spilt_seed=seed, fea=X_nolab, label=Y_nolab, nfold=nfold, fold_num=fold_num)
#     return train_X_nolab, train_Y_nolab, test_X_nolab, test_Y_nolab



def label_extra(data, t_col="Time", e_col="Event"):
    X = data[[c for c in data.columns if c not in [t_col, e_col]]]
    Y = data[[t for t in data.columns if t in[t_col]]]
    Y.loc[data[e_col]==0] = -Y.loc[data[e_col]==0]
    return X.values,Y.values



def read_data(filename):
    train_data = pd.read_csv(filename + "train.csv")
    test_data = pd.read_csv(filename + "test.csv")
    train_X,train_Y = label_extra(train_data)
    test_X, test_Y = label_extra(test_data)
    return train_X, train_Y, test_X, test_Y



def load_omic_data_lab(fea_filename):
    data_fea = pd.read_csv(fea_filename)
    headers = data_fea.columns.values.tolist()
    headers = headers[1:]
    headers = np.array(headers)
    time = data_fea.iloc[0,:].tolist()[1:]
    time = np.array(time)   
    #np.random.shuffle(time)  #随机打散，制造伪标签
    status = data_fea.iloc[1, :].tolist()[1:]
    status = np.array(status)
    #np.random.shuffle(status) #随机打散，制造伪标签
    data_fea = data_fea[2:]  ##delete label
    for i in range(len(time)):
        if status[i] == 0:
            time[i] = -time[i]
    data_fea = data_fea.drop('GeneSymbol', axis=1)
    data_fea = data_fea.values
    data_fea = np.transpose(data_fea)
    data_time = time.reshape(-1, 1)
    #np.random.shuffle(data_time) ##随机打散，制造伪标签
    print(len(data_fea))
    print(len(data_time))
    return data_fea, data_time, headers

def load_omic_data_nolab(fea_filename):
    data_fea = pd.read_csv(fea_filename)
    headers = data_fea.columns.values.tolist()
    headers = headers[1:]
    headers = np.array(headers)
    # time = data_fea.iloc[0,:].tolist()[1:]
    # time = np.array(time)  

   #要注意修改label的数量，本测试label是173
    time_1 = torch.randint(0,100,(1,13))#.t().numpy() #随机生成每一个元素是0或者1的张量,shape[173,1]
    time_2 = torch.randint(100,1000,(1,138))#.t().numpy() #随机生成每一个元素是0或者1的张量,shape[173,1]
    time_3 = torch.randint(1000,2000,(1,15))#.t().numpy() #随机生成每一个元素是0或者1的张量,shape[173,1]
    time_4 = torch.randint(2000,3000,(1,7))#.t().numpy() #随机生成每一个元素是0或者1的张量,shape[173,1]
    W = torch.rand(1,173)*10
    b = torch.rand(1,173)
    Label = torch.cat((time_1,time_2,time_3,time_4),dim=1)*W + b
    time = torch.squeeze(Label).numpy() 
    np.random.shuffle(time)  #随机打散，制造伪标签

    status = torch.randint(0,2,(1,173)).squeeze() 
    #status = data_fea.iloc[0, :].tolist()[1:]
    status = np.array(status)
    np.random.shuffle(status) #随机打散，制造伪标签
    data_fea = data_fea[0:]  ##delete label
    for i in range(len(time)):
        if status[i] == 0:
            time[i] = -time[i]
    data_fea = data_fea.drop('GeneSymbol', axis=1)
    data_fea = data_fea.values
    data_fea = np.transpose(data_fea)
    data_time = time.reshape(-1, 1)
    #np.random.shuffle(data_time) ##随机打散，制造伪标签
    print(len(data_fea))
    print(len(data_time))
    return data_fea, data_time, headers

def load_omic_data_other(fea_filename):
    data_fea = pd.read_csv(fea_filename)
    headers = data_fea.columns.values.tolist()
    headers = headers[1:]
    headers = np.array(headers)
    #time = data_fea.iloc[0,:].tolist()[1:]

    time_1 = torch.randint(0,100,(1,13))#.t().numpy() #随机生成每一个元素是0或者1的张量,shape[173,1]
    time_2 = torch.randint(100,1000,(1,138))#.t().numpy() #随机生成每一个元素是0或者1的张量,shape[173,1]
    time_3 = torch.randint(1000,2000,(1,15))#.t().numpy() #随机生成每一个元素是0或者1的张量,shape[173,1]
    time_4 = torch.randint(2000,3000,(1,7))#.t().numpy() #随机生成每一个元素是0或者1的张量,shape[173,1]
    W = torch.rand(1,173)*10
    b = torch.rand(1,173)
    Label = torch.cat((time_1,time_2,time_3,time_4),dim=1)*W + b
    time = torch.squeeze(Label).numpy() 
    np.random.shuffle(time)  #随机打散，制造伪标签

    status = torch.randint(0,2,(1,173)).squeeze() 
    #status = data_fea.iloc[0, :].tolist()[1:]
    status = np.array(status)
    np.random.shuffle(status) #随机打散，制造伪标签
    data_fea = data_fea[0:]  ##delete label
    for i in range(len(time)):
        if status[i] == 0:
            time[i] = -time[i]
    data_fea = data_fea.drop('GeneSymbol', axis=1)
    data_fea = data_fea.values
    data_fea = np.transpose(data_fea)
    data_time = time.reshape(-1, 1)
    #np.random.shuffle(data_time) ##随机打散，制造伪标签
    print(len(data_fea))
    print(len(data_time))
    return data_fea, data_time, headers


def plot_curve(curve_data, title="train epoch-Cindex curve", x_label="epoch", y_label="Cindex"):
    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(curve_data[0], curve_data[1], color='black', markerfacecolor='black', marker='o', markersize=1)
    plt.show()
    ###print sorted survival time:

def split_data(spilt_seed, fea, label, nfold, fold_num = 0):    #label=time (数据中的time，第二行)
    kf = KFold(n_splits=nfold, shuffle=True, random_state=spilt_seed)
    #kf = KFold(n_splits=nfold)
    ###对齐输入

    label_flat = label.flatten()
    censor_index = np.where(label_flat < 0)  #np.where(condition) 满足条件，返回对应的坐标
    no_cen_index = np.where(label_flat >= 0)  
    censor = label[[tuple(censor_index)]]     #截尾label
    censor_fea = fea[[tuple(censor_index)]]    #截尾特征
    censor = censor[0]

    tensor_sub = torch.from_numpy(censor)
    censor_tensor_squee = torch.squeeze(tensor_sub) 
    censor = censor_tensor_squee.numpy()
    
    censor_fea_sub = censor_fea[0]
    tensor = torch.from_numpy(censor_fea_sub)
    censor_fea_tensor_squee = torch.squeeze(tensor) 
    censor_fea = censor_fea_tensor_squee.numpy()

    #censor_fea = censor_fea[0]
    

    #print("=====censor_fea=====",censor_fea)

    no_cen = label[[tuple(no_cen_index)]]   #no截尾label
    nocen_fea = fea[[tuple(no_cen_index)]]  #no截尾特征

    no_cen = no_cen[0]
    no_cen_tensor = torch.from_numpy(no_cen)
    no_cen_tensor_squee = torch.squeeze(no_cen_tensor) 
    no_cen = no_cen_tensor_squee.numpy()
    
    nocen_fea = nocen_fea[0]
    nocen_fea_tensor = torch.from_numpy(nocen_fea)
    nocen_fea_tensor_squee = torch.squeeze(nocen_fea_tensor) 
    nocen_fea = nocen_fea_tensor_squee.numpy()


    # no_cen = no_cen[0]
    # nocen_fea = nocen_fea[0]
    num = 0
    for train_index, test_index in kf.split(censor_fea):  #censor_fea 截尾数据特征
        train_X1 = censor_fea[train_index]
        train_Y1 = censor[train_index]
        test_X1 = censor_fea[test_index]
        test_Y1 = censor[test_index]
        if num == fold_num:
            break
        num += 1

    num = 0
    for train_index, test_index in kf.split(nocen_fea):  #nocen_fea 没有截尾数据特征
        train_X2 = nocen_fea[train_index]
        train_Y2 = no_cen[train_index]
        test_X2 = nocen_fea[test_index]
        test_Y2 = no_cen[test_index]
        if num == fold_num:
            break
        num += 1

    train_X = np.vstack((train_X1, train_X2))  #竖直拼接一个张量，
    train_Y = np.hstack((train_Y1, train_Y2))
    test_X = np.vstack((test_X1, test_X2))
    test_Y = np.hstack((test_Y1, test_Y2))
    return train_X, train_Y, test_X, test_Y


def split_data_sum(spilt_seed, fea_lab, label_lab, fea_nolab, label_nolab, fea_other, label_other, nfold, fold_num = 0):    #label=time (数据中的time，第二行)
    kf = KFold(n_splits=nfold, shuffle=True, random_state=spilt_seed)
    #kf = StratifiedKFold(n_splits=5)
    #kf = KFold(n_splits=nfold)
    ###对齐输入

    label_flat_lab = label_lab.flatten()
    label_flat_nolab = label_nolab.flatten()
    label_flat_other = label_other.flatten()
    censor_index_lab = np.where(label_flat_lab < 0)  #np.where(condition) 满足条件，返回对应的坐标
    censor_index_nolab = np.where(label_flat_nolab < 0)  #np.where(condition) 满足条件，返回对应的坐标
    censor_index_other = np.where(label_flat_other < 0)  #np.where(condition) 满足条件，返回对应的坐标
    no_cen_index_lab = np.where(label_flat_lab >= 0)  
    no_cen_index_nolab = np.where(label_flat_nolab >= 0)  
    no_cen_index_other = np.where(label_flat_other >= 0)  
    censor_lab = label_lab[[tuple(censor_index_lab)]]     #截尾label
    censor_nolab = label_nolab[[tuple(censor_index_nolab)]]     #截尾label
    censor_other = label_other[[tuple(censor_index_other)]]     #截尾label
    censor_fea_lab = fea_lab[[tuple(censor_index_lab)]]    #截尾特征
    censor_fea_nolab = fea_nolab[[tuple(censor_index_nolab)]]    #截尾特征
    censor_fea_other = fea_other[[tuple(censor_index_other)]]    #截尾特征
    censor_lab = censor_lab[0]
    censor_nolab = censor_nolab[0]
    censor_other = censor_other[0]

    tensor_sub_lab = torch.from_numpy(censor_lab)
    tensor_sub_nolab = torch.from_numpy(censor_nolab)
    tensor_sub_other = torch.from_numpy(censor_other)
    censor_tensor_squee_lab = torch.squeeze(tensor_sub_lab) 
    censor_tensor_squee_nolab = torch.squeeze(tensor_sub_nolab) 
    censor_tensor_squee_other = torch.squeeze(tensor_sub_other) 
    censor_lab = censor_tensor_squee_lab.numpy()
    censor_nolab = censor_tensor_squee_nolab.numpy()
    censor_other = censor_tensor_squee_other.numpy()
    
    censor_fea_sub_lab = censor_fea_lab[0]
    censor_fea_sub_nolab = censor_fea_nolab[0]
    censor_fea_sub_other = censor_fea_other[0]
    tensor_lab = torch.from_numpy(censor_fea_sub_lab)
    tensor_nolab = torch.from_numpy(censor_fea_sub_nolab)
    tensor_other = torch.from_numpy(censor_fea_sub_other)
    censor_fea_tensor_squee_lab = torch.squeeze(tensor_lab) 
    censor_fea_tensor_squee_nolab = torch.squeeze(tensor_nolab) 
    censor_fea_tensor_squee_other = torch.squeeze(tensor_other) 
    censor_fea_lab = censor_fea_tensor_squee_lab.numpy()
    censor_fea_nolab = censor_fea_tensor_squee_nolab.numpy()
    censor_fea_other = censor_fea_tensor_squee_other.numpy()

    #censor_fea = censor_fea[0]
    

    #print("=====censor_fea=====",censor_fea)

    no_cen_lab = label_lab[[tuple(no_cen_index_lab)]]   #no截尾label
    no_cen_nolab = label_nolab[[tuple(no_cen_index_nolab)]]   #no截尾label
    no_cen_other = label_other[[tuple(no_cen_index_other)]]   #no截尾label
    nocen_fea_lab = fea_lab[[tuple(no_cen_index_lab)]]  #no截尾特征
    nocen_fea_nolab = fea_nolab[[tuple(no_cen_index_nolab)]]  #no截尾特征
    nocen_fea_other = fea_other[[tuple(no_cen_index_other)]]  #no截尾特征

    no_cen_lab = no_cen_lab[0]
    no_cen_nolab = no_cen_nolab[0]
    no_cen_other = no_cen_other[0]
    no_cen_tensor_lab = torch.from_numpy(no_cen_lab)
    no_cen_tensor_nolab = torch.from_numpy(no_cen_nolab)
    no_cen_tensor_other = torch.from_numpy(no_cen_other)
    no_cen_tensor_squee_lab = torch.squeeze(no_cen_tensor_lab) 
    no_cen_tensor_squee_nolab = torch.squeeze(no_cen_tensor_nolab) 
    no_cen_tensor_squee_other = torch.squeeze(no_cen_tensor_other) 
    no_cen_lab = no_cen_tensor_squee_lab.numpy()
    no_cen_nolab = no_cen_tensor_squee_nolab.numpy()
    no_cen_other = no_cen_tensor_squee_other.numpy()
    
    nocen_fea_lab = nocen_fea_lab[0]
    nocen_fea_nolab = nocen_fea_nolab[0]
    nocen_fea_other = nocen_fea_other[0]
    nocen_fea_tensor_lab = torch.from_numpy(nocen_fea_lab)
    nocen_fea_tensor_nolab = torch.from_numpy(nocen_fea_nolab)
    nocen_fea_tensor_other = torch.from_numpy(nocen_fea_other)
    nocen_fea_tensor_squee_lab = torch.squeeze(nocen_fea_tensor_lab) 
    nocen_fea_tensor_squee_nolab = torch.squeeze(nocen_fea_tensor_nolab) 
    nocen_fea_tensor_squee_other = torch.squeeze(nocen_fea_tensor_other) 
    nocen_fea_lab = nocen_fea_tensor_squee_lab.numpy()
    nocen_fea_nolab = nocen_fea_tensor_squee_nolab.numpy()
    nocen_fea_other = nocen_fea_tensor_squee_other.numpy()


    # no_cen = no_cen[0]
    # nocen_fea = nocen_fea[0]
    num = 0
    for train_index_lab, test_index_lab in kf.split(censor_fea_lab):  #censor_fea 截尾数据特征
        train_X1_lab = censor_fea_lab[train_index_lab]
        train_Y1_lab = censor_lab[train_index_lab]
        test_X1_lab = censor_fea_lab[test_index_lab]
        test_Y1_lab = censor_lab[test_index_lab]
        if num == fold_num:
            break
        num += 1

    num = 0
    for train_index_nolab, test_index_nolab in kf.split(censor_fea_nolab):  #censor_fea 截尾数据特征
        train_X1_nolab = censor_fea_nolab[train_index_nolab]
        train_Y1_nolab = censor_nolab[train_index_nolab]
        test_X1_nolab = censor_fea_nolab[test_index_nolab]
        test_Y1_nolab = censor_nolab[test_index_nolab]
        if num == fold_num:
            break
        num += 1

    num = 0
    for train_index_other, test_index_other in kf.split(censor_fea_other):  #censor_fea 截尾数据特征
        train_X1_other = censor_fea_other[train_index_other]
        train_Y1_other = censor_other[train_index_other]
        test_X1_other = censor_fea_other[test_index_other]
        test_Y1_other = censor_other[test_index_other]
        if num == fold_num:
            break
        num += 1

    num = 0
    for train_index_lab, test_index_lab in kf.split(nocen_fea_lab):  #nocen_fea 没有截尾数据特征
        train_X2_lab = nocen_fea_lab[train_index_lab]
        train_Y2_lab = no_cen_lab[train_index_lab]
        test_X2_lab = nocen_fea_lab[test_index_lab]
        test_Y2_lab = no_cen_lab[test_index_lab]
        if num == fold_num:
            break
        num += 1

    num = 0
    for train_index_nolab, test_index_nolab in kf.split(nocen_fea_nolab):  #nocen_fea 没有截尾数据特征
        train_X2_nolab = nocen_fea_nolab[train_index_nolab]
        train_Y2_nolab = no_cen_nolab[train_index_nolab]
        test_X2_nolab = nocen_fea_nolab[test_index_nolab]
        test_Y2_nolab = no_cen_nolab[test_index_nolab]
        if num == fold_num:
            break
        num += 1

    num = 0
    for train_index_other, test_index_other in kf.split(nocen_fea_other):  #nocen_fea 没有截尾数据特征
        train_X2_other = nocen_fea_other[train_index_other]
        train_Y2_other = no_cen_other[train_index_other]
        test_X2_other = nocen_fea_other[test_index_other]
        test_Y2_other = no_cen_other[test_index_other]
        if num == fold_num:
            break
        num += 1


    train_X_lab = np.vstack((train_X1_lab, train_X2_lab))  #竖直拼接一个张量，
    train_X_nolab = np.vstack((train_X1_nolab, train_X2_nolab))  #竖直拼接一个张量，
    train_X_other = np.vstack((train_X1_other, train_X2_other))  #竖直拼接一个张量，
    train_Y_lab = np.hstack((train_Y1_lab, train_Y2_lab))
    train_Y_nolab = np.hstack((train_Y1_nolab, train_Y2_nolab))
    train_Y_other = np.hstack((train_Y1_other, train_Y2_other))
    test_X_lab = np.vstack((test_X1_lab, test_X2_lab))
    test_X_nolab = np.vstack((test_X1_nolab, test_X2_nolab))
    test_X_other = np.vstack((test_X1_other, test_X2_other))
    test_Y_lab = np.hstack((test_Y1_lab, test_Y2_lab))
    test_Y_nolab = np.hstack((test_Y1_nolab, test_Y2_nolab))
    test_Y_other = np.hstack((test_Y1_other, test_Y2_other))
    return train_X_lab, train_Y_lab, test_X_lab, test_Y_lab, train_X_nolab, train_Y_nolab, test_X_nolab, test_Y_nolab, train_X_other, train_Y_other, test_X_other, test_Y_other


def split_data_with_headers(spilt_seed, fea, label,headers, nfold, fold_num = 0):
    kf = KFold(n_splits=nfold, shuffle=True, random_state=spilt_seed)
    ###对齐输入

    label_flat = label.flatten()
    censor_index = np.where(label_flat < 0)
    no_cen_index = np.where(label_flat >= 0)
    censor = label[[tuple(censor_index)]]
    censor_fea = fea[[tuple(censor_index)]]
    censor_headers = headers[[tuple(censor_index)]]
    censor = censor[0]
    censor_fea = censor_fea[0]
    censor_headers = censor_headers[0]
    censor_headers = censor_headers.reshape(((len(censor_headers), 1)))  # (3,1)

    no_cen = label[[tuple(no_cen_index)]]
    nocen_fea = fea[[tuple(no_cen_index)]]
    no_cen_headers = headers[[tuple(no_cen_index)]]

    no_cen = no_cen[0]
    nocen_fea = nocen_fea[0]
    no_cen_headers = no_cen_headers[0]
    no_cen_headers = no_cen_headers.reshape(((len(no_cen_headers), 1)))  # (3,1)



    num = 0
    for train_index, test_index in kf.split(censor_fea):
        train_X1 = censor_fea[train_index]
        train_Y1 = censor[train_index]
        train_headers1 = censor_headers[train_index]
        test_X1 = censor_fea[test_index]
        test_Y1 = censor[test_index]
        test_headers1 = censor_headers[test_index]
        if num == fold_num:
            break
        num +=1

    num = 0
    for train_index, test_index in kf.split(nocen_fea):
        train_X2 = nocen_fea[train_index]
        train_Y2 = no_cen[train_index]
        train_headers2 = no_cen_headers[train_index]
        test_X2 = nocen_fea[test_index]
        test_Y2 = no_cen[test_index]
        test_headers2 = no_cen_headers[test_index]
        if num == fold_num:
            break
        num +=1

    train_X = np.vstack((train_X1, train_X2))
    train_Y = np.vstack((train_Y1, train_Y2))
    train_headers = np.vstack((train_headers1,train_headers2))
    test_X = np.vstack((test_X1, test_X2))
    test_Y = np.vstack((test_Y1, test_Y2))
    test_headers = np.vstack((test_headers1,test_headers2))
    return train_X, train_Y, test_X, test_Y,train_headers, test_headers

# def split_data(spilt_seed, fea, label,headers, nfold=5, fold_num=0):
#     kf = KFold(n_splits=nfold, shuffle=True, random_state=spilt_seed)
#     train_X = []
#     train_Y = []
#     test_X = []
#     test_Y = []
#     num = 0
#     for train_index, test_index in kf.split(fea):
#         train_X = fea[train_index]
#         train_Y = label[train_index]
#         train_headers = headers[train_index]
#         test_X = fea[test_index]
#         test_Y = label[test_index]
#         test_headers=headers[test_index]
#         if num == fold_num:
#             print(num)
#             break
#         num += 1
#     train_X, train_Y, train_headers = sort_surv_data(train_X, train_Y, train_headers)
#     test_X, test_Y, test_headers= sort_surv_data(test_X, test_Y, test_headers)
#     return train_X, train_Y, test_X, test_Y,train_headers, test_headers