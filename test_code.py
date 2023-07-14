import torch
import numpy as np
import csv
import pandas

L_Y = torch.rand(5,1).t().numpy()                 #.numpy()       #.astype(np.float32)
L_int = torch.randint(0,2,(5,1)).t().numpy() #随机生成每一个元素是0或者1的张量,shape[173,1]


data = [
    ['Name', 'Age', 'Gender'],
    ['Tom', '18', 'Male'],
    ['Lily', '20', 'Female'],
    ['Tina', '21', 'Female']
]

lab = {'time':[2,2]}

# 找到要修改的位置
row, col = 4, 3

# 修改指定位置的数据
data[row-1][col-1] = L_Y

with open('.\my_dataset\example.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(L_int)
    writer.writerows(L_Y)
    writer.writerows(data)

df = pandas.DataFrame(L_Y)
df.to_csv('.\my_dataset\example.csv',mode='lab',index=6,header=False) #mode为追加的数据；index为每一行的索引序号；header为标题
# print("=====L_Y====",L_Y)
# print("=====L_int====",L_int.squeeze(1))