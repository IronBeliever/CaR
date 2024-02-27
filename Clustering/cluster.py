from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from jsonargparse import ArgumentParser
from sklearn.decomposition import PCA
import math
import time
import json
import warnings
warnings.filterwarnings('ignore')


parser = ArgumentParser(description="Command for sort instruction tuning dataset by COMET score.")
parser.add_argument(
    "--input",
    help=(
        "Path to the directory where intruction dataset will be stored. "
        + "By default its saved in ./data/IQS_ranking_result.json"
    ),
    default=None,
)
cfg = parser.parse_args()

if cfg.input is not None:
    with open(cfg.input, "r") as f:
        origent_data = json.load(f) 
else:
    with open("../data/IQS_ranking_result.json", "r") as f:
        origent_data = json.load(f) 

'''
    tokenize
'''
# Load model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence_bert')

vec_x_v1 = []
i=0

for item in origent_data:
    instruction = "Instruction: "+ item["instruction"] + ' Input: ' + item["input"] + ' Response: ' + item["output"]
    i+=1
    output = model.encode(instruction)
    # print(output.shape)     # 384  (768/2)
    vec_x_v1.append(output)
    # if i % 1000 == 0:
    #     print(time.time())
    #     print("{}k / 52k".format(i/1000))
    #     print('------------------')


print("\n--------------Dimensional Analysis----------------")

print(len(vec_x_v1))
print(np.shape(vec_x_v1[0]))

X_vec = np.array(vec_x_v1)
print(type(X_vec))

# # Save intermediate results
# np.savetxt("sentence_vector.txt", X_vec, fmt='%f', delimiter=',')
#
#X_vec = np.loadtxt("sentence_vector.txt", dtype='float', comments='#', delimiter=',')


'''
    Principal Component Analysis
    n_components: Number of principal components retained, percentage
'''
pca = PCA(n_components=0.95, random_state=42)
X_reduced = pca.fit_transform(X_vec)
print(X_reduced.shape)      # (52002, 243)


'''
    run k-means on the feature vectors processed by PCA
'''
k = int(math.sqrt(52002 / 2))
kmeans = KMeans(n_clusters=k, random_state=42)  
labels_pred = kmeans.fit_predict(X_reduced) 

i = 0
for item in origent_data:
    item['label'] = str(labels_pred[i])
    i += 1

json.dump(origent_data, open('../data/Clustering_result.json', 'w'))  
