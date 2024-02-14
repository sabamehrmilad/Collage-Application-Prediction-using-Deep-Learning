#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


# In[2]:


data = pd.read_csv('D:\\Data science\\project\\college.csv')
data


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.info()


# In[6]:


data1=data.iloc[:,1:]
data1


# In[7]:


dummies = pd.get_dummies(data1[['Private']])
dummies.head() 


# In[8]:


X_ = data1.drop(['Private'], axis=1)
main_data = pd.concat([X_, dummies['Private_Yes']], axis = 1)


# In[9]:


main_data.head()


# In[10]:


#Split Data into Train and Test 
from sklearn.model_selection import train_test_split 
train, test = train_test_split(main_data, test_size = 0.2, random_state = 1234)


# In[11]:


train.shape


# In[12]:


test.shape


# In[13]:


train_targets = train['Apps']
train = train.drop(['Apps'], axis=1)
train_targets.head() 


# In[14]:


train_targets.head()


# In[15]:


test.head()


# In[16]:


test_targets = test['Apps']
test = test.drop(['Apps'], axis=1)
test_targets 


# # Data Preprocessing 
Normalizing the Data 
# In[17]:


mean = train.mean(axis=0)
train -= mean 
std = train.std(axis=0)
train /= std 
test -= mean 
test /= std 


# # Building your model 

# model definition 

# In[18]:


from tensorflow import keras
from tensorflow.keras import layers


# In[19]:


train.shape


# Version A : As a list of layers:

# In[20]:


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation = "relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer = "rmsprop", loss="mae", metrics=["mae"])
    return model

Version B : By adding the layers explicitly:
# In[21]:


def build_model():
    model = keras.models.Sequential()
    model.add(layers.InputLayer(input_shape=(train.shape[1],)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(optimizer="rmsprop", loss="mae", metrics=["mae"])
    return model


# In[22]:


model = build_model()
model.summary()


# In[23]:


#Configure the model 
model.compile(optimizer = "rmsprop", loss="mae", metrics=["mae"])


# In[24]:


model.fit(train, train_targets, epochs=100, batch_size = 125)


# In[25]:


pred_ann = model.predict(test)
pred_ann = pd.Series(pred_ann[:, 0], index = test_targets.index)
pred_ann.head()


# In[26]:


#batch_size : 250 , epochs = 200
model.fit(train, train_targets, epochs=200, batch_size = 250)


# In[27]:


pred_ann2 = model.predict(test)
pred_ann2 = pd.Series(pred_ann2[:, 0], index = test_targets.index)
pred_ann.head()


# # Validating your approach using K-fold validation 

# In[28]:


import numpy as np
k = 4
num_val_samples = len(train) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train = np.concatenate(
        [train[:i * num_val_samples],
         train[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    model.fit(partial_train, partial_train_targets,
              epochs=num_epochs, batch_size=16, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)


# In[29]:


all_scores


# In[30]:


np.mean(all_scores)


# In[31]:


num_epochs = 500
all_mae_histories = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train[:i * num_val_samples],
         train[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=16, verbose=0)
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)


# In[32]:


len(all_mae_histories)


# In[33]:


len(all_mae_histories[0])


# # Building the history of successive mean k-fold scores

# In[34]:


average_mae_history =[
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


# In[35]:


import matplotlib.pyplot as plt 
plt.plot(range(1, len(average_mae_history)+1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()


# In[37]:


model = build_model()
model.fit(train, train_targets , epochs=250, batch_size= 64, verbose=0)  
test_mse_score, test_mae_score  = model.evaluate(test, test_targets)


# In[38]:


test_mae_score 


# In[39]:


predictions = model.predict(test)
predictions[0]


# In[40]:


predictions 


# In[ ]:




