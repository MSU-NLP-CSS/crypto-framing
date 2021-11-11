# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import transformers as ppb # pytorch transformers

# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


# %%
import pandas as pd
df = pd.read_csv("./Dataset_with_cosine.csv")
from sklearn.utils import shuffle
df = shuffle(df)


# %%
df


# %%
tokenized = df['tweet'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


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
labels = df['buy_sell']


# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)


from keras.utils import to_categorical
labels = to_categorical(labels)


# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)


# %%
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional, Input
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report

def rnn( features, labels,  X_test, Y_test, shape,loss_func, metrics, batch_size, epochs, split ):
    inputs = Input(shape = (shape, ) )
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.001)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.001)(x)
    x = Dense(3, activation='softmax')(x)
    model = Model( inputs, x)

    
    model.compile(loss=loss_func, optimizer='nadam', metrics=[metrics])
    model.fit(features, np.array(labels), batch_size=batch_size, validation_split=split, epochs=epochs)

    

    y_pred = model.predict(X_test, batch_size=16, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)

    return y_pred, y_pred_bool
  


# %%
y_pred, y_pred_bool = rnn(X_train, X_test, y_train, y_test, 768, 'categorical_crossentropy', 'accuracy', 16, 49, 0.1 )
y_test_bool = np.argmax(Y_test, axis=1)
print(y_test_bool)
from sklearn.metrics import classification_report
print(classification_report(y_test_bool, y_pred_bool))

# %% [markdown]
# ### Random Forest

# %%
## my custom functions 
import models


labels = df['buy_sell']
randomForest = models.random_forest( features, labels )


