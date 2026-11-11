import pandas as pd
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
from munkres import Munkres, print_matrix


# 利用真实标签中重复度最大的值直接确定标签映射关系:
# 标签匹配  前提:确保各元素一一对应
# idea: 每一簇的真实标签赋值为该簇中元素所属真实标签数目最多的标签项
# input: true,cluster  output: cluster_map
def cluster_map_true(cluster_array, label_array):
	cluster_map = np.array([0 for index in range(len(cluster_array))])  # map后的cluster
	for elements in np.unique(cluster_array):
		cluster_temp_index = np.where(cluster_array == elements)[0]  # 每个cluster对应的index
		cluster_temp_true = label_array[cluster_temp_index]  # 通过index找到每簇cell对应的真实标签
		count = np.bincount(cluster_temp_true)  # 因np.bincount统计0~max的次数,argmax返回最大数的下标/索引,故返回的索引即簇中出现次数最多的真实label
		cluster_map[cluster_temp_index] = np.argmax(count)  # 找到出现次数最多的true标签赋值给该簇
	return cluster_map


# 使用匈牙利算法,完成标签的映射
def best_map(L1, L2):
	# L1 should be the labels and L2 should be the clustering number we got
	Label1 = np.unique(L1)  # 去除重复的元素，由小大大排列
	nClass1 = len(Label1)  # 标签的大小
	Label2 = np.unique(L2)
	nClass2 = len(Label2)
	nClass = np.maximum(nClass1, nClass2)
	G = np.zeros((nClass, nClass))
	for i in range(nClass1):
		ind_cla1 = L1 == Label1[i]
		ind_cla1 = ind_cla1.astype(float)
		for j in range(nClass2):
			ind_cla2 = L2 == Label2[j]
			ind_cla2 = ind_cla2.astype(float)
			G[i, j] = np.sum(ind_cla2 * ind_cla1)
	m = Munkres()
	index = m.compute(-G.T)
	index = np.array(index)
	c = index[:, 1]
	newL2 = np.zeros(L2.shape)
	for i in range(nClass2):
		newL2[L2 == Label2[i]] = Label1[c[i]]
	return newL2


# read the file
# 声明:这两个文件有头文件
true = pd.read_csv(r"E:\SC3实验\Chung\Chung_sc3_label.csv").values  # 此处可以转化为字符串的拼接，拼接出路径，更通用，便捷

# 字符串替换
unique = np.unique(true[:, 1])
print(unique)
# 将cell类型用具体的数值代替
# 1. 循环+if判断  2.for循环，判断元素位于unique中数组位置(更通用)
for i in range(len(true)):
	true[i, -1] = int(np.where(unique == true[i, -1])[0]) + 1
# 个人认为偏好标签从1开始计算，而数组计数从0开始，故＋1，当然可不加，但保险起见，建议+1
true = true[:, -1].astype(int)


cluster = pd.read_csv(r'E:\SC3实验\Chung\Chung_sc3_Cluster.csv').values
cluster = cluster[:, -1].astype(int)

# 真实标签与聚类结果映射:
cluster_map = cluster_map_true(cluster, true)  # max重复度
cluster_Munkres_map = best_map(true, cluster).astype(int)  # 匈牙利算法

# calculate ARI & NMI
ARI = adjusted_rand_score(true, cluster_map)
NMI = metrics.normalized_mutual_info_score(true, cluster_map)

print('#######################')
print('ARI {}'.format(ARI))
print('NMI {}'.format(NMI))