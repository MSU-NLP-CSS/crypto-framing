# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import transformers as ppb # pytorch transformers


## Want BERT instead of distilBERT? Uncomment the following line:
model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


# %%
model


# %%
import pandas as pd
df = pd.read_csv("./Filtered_Small_Bert_cosine.csv")

from sklearn.utils import shuffle
df = shuffle(df)

tokenized = df['tweet'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


# %%
print(df)


# %%
import numpy as np

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])


# %%
np.array(padded).shape
attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape


# %%
input_ids = torch.tensor(padded).to(torch.int64)
print( input_ids) 
attention_mask = torch.tensor(attention_mask)
print( attention_mask )
with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)
features = last_hidden_states[0][:,0,:].numpy()


# %%
from sklearn.preprocessing import LabelEncoder

labels = df['buy_sell']
le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)


from keras.utils import to_categorical
labels = to_categorical(labels)


# %%
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional, Input
from keras.models import Model


def rnn( features, labels,X_test, Y_test, shape,loss_func, metrics, batch_size, epochs, split ):
    inputs = Input(shape = (shape, ) )
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.001)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.001)(x)
    x = Dense(3, activation='softmax')(x)
    model = Model( inputs, x)
    model.compile(loss=loss_func, optimizer='nadam', metrics=[metrics])
    model.fit(features, np.array(labels), batch_size=batch_size, validation_split=split, epochs=epochs)
   
    from sklearn.metrics import classification_report

    y_pred = model.predict(X_test, batch_size=16, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
     

   
    return y_pred, y_pred_bool


# %%
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(features, labels,  test_size=0.2, random_state=42)


# %%
y_pred, y_pred_bool = rnn(X_train, Y_train, X_test, Y_test,  768, 'categorical_crossentropy', 'accuracy', 16, 50, 0.2 )


# %%
y_test_bool = np.argmax(Y_test, axis=1)
print(y_test_bool)


# %%
## BERT COSINE
from sklearn.metrics import classification_report
print(classification_report(y_test_bool, y_pred_bool))


# %%
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, Y_test, verbose=0)

# %% [markdown]
# # Random Forest and Naive Bayes using the bert cosine dataset with BERT features 
# 

# %%
import models
labels = df['buy_sell']
randomForest = models.random_forest( features, labels )


# %%
models.naive_bayes( features, labels )


# %%
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional, Input
from keras.layers.embeddings import Embedding

def bisltm( features, featuresSize, labels, shape,loss_func, metrics, optimizer, batch_size, epochs, split ):
    inputs = Input(shape = (shape, ) )
   # x = Embedding( featuresSize, 32, input_length= shape)(inputs)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(3, activation='softmax')(x)
    model = Model( inputs, x)
    model.compile(loss=loss_func, optimizer=optimizer, metrics=[metrics])
    model.fit(features, np.array(labels), batch_size=batch_size, validation_split=split, epochs=epochs)


# %%
rnn(features, labels, max_len, 768, 'categorical_crossentropy', 'accuracy', 16, 50, 0.2 )


# %%
print(max_len)


# %%
print(features.shape)


# %%
test = np.reshape( features, (features.shape[0], 1, features.shape[1]))


# %%
print(test.shape)


# %%
from keras.models import Sequential
model = Sequential()
model.add(LSTM(64, input_shape=( 1, 768, )))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
print(model.summary())
model.fit(test, labels, epochs=50, batch_size=16, validation_split=0.2)


# %%
from keras.models import Sequential
model = Sequential()
model.add(Bidirectional( LSTM(64, input_shape=( 1, 768, ))))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
print(model.summary())
model.fit(test, labels, epochs=50, batch_size=16, validation_split=0.2)


# %%
bisltm(features, 30000, labels, 768, 'categorical_crossentropy','accuracy', 'nadam', 16, 50, 0.2)


# %%
# Orignial LDA dataset, without BERT cosine distance , using Count vectorizer 


# %%
### Count vectorizer 

import pandas as pd
df = pd.read_csv("./LDA_neutral.csv")

from sklearn.utils import shuffle
df = shuffle(df)


# %%
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
features = vectorizer.fit_transform(df['tweet'])
labels = df['buy_sell']


# %%
randomForest = models.random_forest( features, labels )


# %%
models.naive_bayes( features.toarray(), labels )


# %%



