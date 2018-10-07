import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[44]:


df = pd.read_csv('./coinbaseUSD_1-min_data_2014-12-01_to_2018-06-27.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
df.set_index('Timestamp', inplace=True)
df.head()


# In[45]:


df.tail()


# In[105]:


group = df.groupby(df.index)
Real_Price = group['Weighted_Price'].mean()


# In[106]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
Real_Price = sc.fit_transform(Real_Price.values.reshape(-1,1))


# In[107]:


Real_Price.shape


# In[108]:


X, Y = list(), list()
for i in range(0,len(Real_Price)-1-(len(Real_Price)%60)):
    X.append(Real_Price[i:i+30])
    Y.append(Real_Price[i+30:i+60])
len(X)
len(Y)


# In[109]:


num_train = int(len(Y) * 0.7)
num_train


# In[110]:


train_x = X[0:num_train]
train_y = Y[0:num_train]

test_x = X[num_train:]
test_y = Y[num_train:]


# In[111]:


test_y[0].shape


# In[125]:


train_x = np.reshape(train_x, (len(train_x), 30, 1 ))
train_y = np.array(train_y)


# In[126]:


print(train_x.shape)
print(train_y.shape)


# In[127]:


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector


# In[ ]:


# Initialising the RNN
model = Sequential()

# Adding the input layer and the LSTM layer
model.add(LSTM(batch_input_shape=(1,30,1), units = 30, activation = 'sigmoid',return_sequences=False))
model.add(RepeatVector(30))
model.add(LSTM(units = 30, activation = 'sigmoid',return_sequences=True))

# Adding the output layer
model.add(TimeDistributed(Dense(1)))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(train_x, train_y, batch_size = 1, epochs = 100)

model.save('my_model.h5')
