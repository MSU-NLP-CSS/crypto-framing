# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas as pd 
df = pd.read_csv("\\Data\\All Tweets Before Feb 2020\\CSV\\Jan_2017_March_2020.csv")

# %%
df.drop_duplicates("tweet", inplace=True)

# %%
import re 

data = df['tweet']
data.dropna(inplace=True)
data = [(i.lower()) for i in data]

### remove emails and @users
data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data]

# Remove https:// links 
data = [re.sub(r'\S*https://\S*\s?', '', sent) for sent in data]


# Remove new line characters
data = [re.sub(r'\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub(r"\'", "", sent) for sent in data]

# %%
from sklearn.feature_extraction.text import CountVectorizer
#vectorizer = CountVectorizer()
vectorizer = CountVectorizer(analyzer='word', min_df=10,
# minimum reqd occurences of a word 
                             stop_words='english',             
# remove stop words
                             lowercase=True,                   
# convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  
# num chars > 3
                             # max_features=50000,             
# max number of uniq words   
                            )
X = vectorizer.fit_transform(data)

#%% 
import numpy as np 
print(X.shape)
indices = np.arange(X.shape[0])
# %%
print( indices )
# %%

from sklearn.model_selection import train_test_split 

X_train, X_test, idx_train, idx_test = train_test_split( X, indices, test_size=0.33, random_state=42 )

# %%
print( X_train.shape[0] )

# %%
from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(X_train.shape[1],))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(X_train.shape[1], activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %% 
X_train.sort_indices()
X_test.sort_indices()
# %%
from keras.callbacks import TensorBoard


autoencoder.fit(X_train, X_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]
                )

autoencoder.save_weights('./ae_weights_repro.h5')

 # %% [markdown]
# ### Cluster 
# 
# %% [markdown]
# code for REFERENCE: https://github.com/hadifar/stc_clustering/blob/master/STC.py
# 

autoencoder.load_weights(".\\ae_weights_100_epochs.h5")

# %%
encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)


# %%
from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans

# %%
print(X_train.shape)
# %%
n_clusters = 10
x= X_train
x = x.reshape((x.shape[0], -1))

# %% 
print(x.shape)
# %%
kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
y_pred_kmeans = kmeans.fit_predict(x)


# %%
from matplotlib import pyplot as plt 
import numpy as np  

print( y_pred_kmeans )
plt.hist(y_pred_kmeans)

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.decomposition import TruncatedSVD

pca_num_components = 2
reduced_data = TruncatedSVD(n_components=pca_num_components, n_iter=7,  random_state=42 ).fit_transform(x)
results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])
sns.scatterplot(x="pca1", y="pca2", hue=y_pred_kmeans, data=results)
plt.title('K-means Clustering with 2 dimensions')
plt.show()

# %% [markdown]
# ## Build clustering model
sns.scatterplot(x="pca1", y="pca2", hue=y_pred_kmeans, data=results)
plt.title('K-means Clustering with 2 dimensions')
plt.show()
sn.savefig('kmeans10Custer.png',  dpi=100)
# %%
class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(name='clusters', shape=(self.n_clusters, input_dim), initializer='glorot_uniform') 
        
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) 
        
        return q
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters
    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# %%
clustering_layer = ClusteringLayer(n_clusters, name='clusters')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)
model.compile(optimizer=SGD(0.01, 0.9), loss='kld')


# %%
kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(x))


# %%

y_pred_last = np.copy(y_pred)


# %%
model.get_layer(name='clusters').set_weights([kmeans.cluster_centers_])

# %% [markdown]
# ### Step 2: deep clustering
# Compute p_i by first raising q_i to the second power and then normalizing by frequency per cluster:

# %%
# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


# %%
loss = 0
index = 0
maxiter = 8000
update_interval = 140
index_array = np.arange(x.shape[0])
tol = 0.001 # tolerance threshold to stop training

# %%
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
batch_size = 100 
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(x, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p

        # check stop criterion - model convergence
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
    print( x.shape )
    print( p.shape)
    x_array = x[idx[0]:idx[-1]]
    y_array = p[idx[0]:idx[-1]]
    print( x_array.shape )
    print( y_array.shape)
    loss = model.fit(x_array, y_array )
    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

# %% [markdown]
model.save_weights('./old_tweets_DEC_model_final_100_epochs.h5')

# %%

model.load_weights(".\\COLING\DEC_model_final.h5")

# %%

q = model.predict(x, verbose=0)
# %%
p = target_distribution(q) 

# %%
y_pred = q.argmax(1)

# %%
print(y_pred[0])

# %%
y_merged = []

for i in range(0,len(y_pred)):
    if y_pred[i]==0 or y_pred[i]==6 or y_pred[i]==9:
        y_merged.append(0)
    elif y_pred[i]==8 or  y_pred[i]==1 or  y_pred[i]==4 or  y_pred[i]==7 or y_pred[i]==5:
        y_merged.append(1)
    else:
        y_merged.append(y_pred[i])

# %%
print( y_merged)
# %%
from matplotlib import pyplot as plt 
import numpy as np  
plt.hist(y_pred)

# %%

sns.scatterplot(x="pca1", y="pca2", hue=y_pred, data=results)
plt.title('K-means Clustering with 2 dimensions')
plt.show()


# %%
unique, counts = np.unique(y_pred, return_counts=True)

print(np.asarray((unique, counts)).T)

# %%


# %%
def GetClusterWordCount( cluster_number, y_pred, x, vectorizer  ):
    bigCluster = []
    for i in range(0, len(y_pred)):
        if y_pred[i] == cluster_number :
            bigCluster.append(x[i])

    sentences = []
    for sent in bigCluster:
        sentences.append(vectorizer.inverse_transform( sent ))

    list = []
    for arr in sentences:
        list.append(arr[0].tolist())

    dictionary = {}

    for sentence in list:
        for word in sentence:
            if word not in dictionary.keys():
                dictionary[word] = 1 
            else:
                dictionary[word] = dictionary[word] + 1

    return sorted(dictionary.items(), key=lambda x: x[1],reverse=True ), len(bigCluster)


# %%
from pandas import DataFrame
deepCluster = DataFrame()

for i in range(0, 10):
    print("cluster: " + str(i))
    cluster, count = GetClusterWordCount( i, y_merged, x, vectorizer)
    clusterDf = DataFrame( cluster, columns=['words','count'])
    word = "Cluster " + str(i) + "Words"
    count = "Cluster " + str(i) +"Word Count"
    deepCluster[word] = clusterDf['words']
    deepCluster[count] = clusterDf['count']


# %%
deepCluster
# %%
deepCluster.to_csv(r'./second_merged_covid_clusters_word_count.csv', index = False)

# %%
print( idx_train.shape )
inverse = vectorizer.inverse_transform( X )

# %%
inverse[0][0]

# %%
df 

# %%
print( X.shape)
# %%
cluster_val = np.full((X.shape[0] + 1, 1), 999 )
print( cluster_val )

for i in range(0, len(y_pred)):
    cluster_val[idx_train[i]] = y_pred[i]

# %%
print( cluster_val.shape )
# %%
df['clusters'] = cluster_val

# %%
df

# %% 
csv_file_path = ".\\Data\\DeepClustering_old_tweets_before_covid_kmeans_10_labeled_cluster_sorted_words.csv"
df.to_csv( csv_file_path, index=False, header=True)

# %%
