from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import math
import time
import json
import warnings
warnings.filterwarnings('ignore')

with open("IQS_ranking_result.json", "r") as f:
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
     if i % 100 == 0:
         print(time.time())
         print("{}/520".format(i/100))
         print('------------------')


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

json.dump(origent_data, open('IQS_Clustering_result.json', 'w'))  
