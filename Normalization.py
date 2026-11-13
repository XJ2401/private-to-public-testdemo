import pandas as pd
import numpy as np


# functions: normalization
def normalization(features_):  # row: cells  column:genes
	features = features_.copy()
	for i in range(len(features)):
		features[i] = features[i] / sum(features[i]) * 100000
	features = np.log2(features + 1)
	return features


data_path = r"E:\实验材料\Chung\data.tsv"  # QC之后的文件
save_path = r'E:\实验材料\Chung\data_norm.tsv'
# Get data
data = pd.read_csv(data_path, index_col=0, header=0, delimiter="\t")
cells = data.columns.values
genes = data.index.values
features = data.values.T
# Preprocess features
features = normalization(features)  # row: cells  column:genes
features = (features - np.mean(features)) / (np.std(features))
print("File is writing...")
data_norm = pd.DataFrame(features.T, index=genes, columns=cells)
data_norm.to_csv(save_path, sep='\t')
print("Writing is finished!")
print("Done")