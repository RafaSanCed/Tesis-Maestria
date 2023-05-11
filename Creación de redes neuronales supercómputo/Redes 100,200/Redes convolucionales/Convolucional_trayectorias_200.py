#!/usr/bin/env python
# coding: utf-8

# # Importamos librerías

# In[1]:


import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras


# In[2]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.initializers import HeNormal
from keras.regularizers import l1, l2


# In[3]:


tf.test.is_gpu_available()


# # Importamos los datos

# In[13]:


path = '../../../Base de datos/Trayectorias 200 eval'


# ## Datos de entrenamiento

# ### Datos

# In[14]:


varftrain = np.load(path + '/varftrain.npy')
varftrain = np.transpose(varftrain, (0,3,1,2))


# In[15]:


np.shape(varftrain)


# ### Coeficientes

# In[16]:


coefvarftrain = np.load(path + '/coefvarftrain.npy')


# In[17]:


np.shape(coefvarftrain)


# ## Datos de prueba

# ### Datos

# In[18]:


varftest = np.load(path + '/varftest.npy')
varftest = np.transpose(varftest, (0,3,1,2))


# In[19]:


np.shape(varftest)


# ### Coeficientes

# In[20]:


coefvarftest = np.load(path + '/coefvarftest.npy')


# In[21]:


np.shape(coefvarftest)


# ## Datos de validación

# ### Datos

# In[22]:


varfval = np.load(path + '/varfval.npy')
varfval = np.transpose(varfval, (0,3,1,2))


# In[23]:


np.shape(varfval)


# ### Coeficienes

# In[24]:


coefvarfval = np.load(path + '/coefvarfval.npy')


# In[25]:


np.shape(coefvarfval)


# # Unimos coeficientes con los datos

# In[26]:


train_dataset = tf.data.Dataset.from_tensor_slices((varftrain, coefvarftrain))
test_dataset = tf.data.Dataset.from_tensor_slices((varftest, coefvarftest))
val_dataset = tf.data.Dataset.from_tensor_slices((varfval, coefvarfval))


# In[27]:


train_dataset


# # Mezclar y procesar por lotes los conjuntos de datos

# In[28]:


BATCH_SIZE = 10
SHUFFLE_BUFFER_SIZE = 10

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)


# # Creación del modelo

# In[29]:


def custom_activation(x):
    return 10 * tf.nn.sigmoid(x)


# In[30]:


model = Sequential()
# Primera capa convolucional
model.add(Conv2D(32, (3, 3), activation='tanh', input_shape=(200, 10, 2),kernel_initializer=HeNormal(), kernel_regularizer=l1(0.001)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Segunda capa convolucional
model.add(Conv2D(64, (3, 3), activation='tanh',kernel_initializer=HeNormal(), kernel_regularizer=l1(0.001) ))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Tercera capa convolucional
#model.add(Conv2D(64, (3, 3), activation='tanh',kernel_initializer=HeNormal(), kernel_regularizer=l1(0.001)))

model.add(Flatten())
model.add(Dense(128, activation='tanh',kernel_initializer=HeNormal(), kernel_regularizer=l1(0.001)))
model.add(Dense(64, activation='tanh',kernel_initializer=HeNormal(), kernel_regularizer=l1(0.001)))
model.add(Dense(64,activation = 'relu',kernel_initializer=HeNormal(), kernel_regularizer=l1(0.001)))
model.add(Dropout(0.3))
model.add(Dense(7, activation= custom_activation))



# In[31]:


model.summary()


# In[32]:


keras.utils.plot_model(model,show_shapes=True)


# # Entrenamos el modelo

# In[33]:


from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001,clipvalue=10.0)


# In[34]:


model.compile(optimizer= optimizer,
              loss= 'mse',
              metrics=['accuracy'])


# In[35]:


def scheduler(epoch, lr):
  if epoch < 100:
    return lr
  else:
    return 1e-3 -  2e-8* epoch


# In[36]:


val_epochs = 20

early_stop = tf.keras.callbacks.EarlyStopping( monitor = 'val_loss', patience = 800,verbose = 1, 
                                              restore_best_weights = True)

reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(train_dataset, validation_data=val_dataset, epochs= val_epochs, callbacks=[reduce_lr, early_stop])


# # Analizamos accuracy y loss

# In[37]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(val_epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# # Guardamos los datos de Accuracy y Loss

# In[38]:


df = pd.DataFrame.from_dict(history.history)
df.to_csv('../../../Gráficas finales/History 100,200/historyconvtrayec.csv', index=False)


# # Guardamos el modelo

# In[39]:


path_to_save = '../../../Modelos/Modelos 100,200/'


# In[40]:


model.save(path_to_save + '/convolucionaltr.h5')


# # Importamos el modelo

# In[41]:


custom_objects = {'custom_activation': custom_activation}
new_model = keras.models.load_model('../../../Modelos/Modelos 100,200/convolucionaltr.h5', custom_objects=custom_objects)


# # Probamos el modelo con datos nuevos

# ## Creamos nuevos datos

# In[58]:


import random
from scipy.integrate import solve_ivp

varf = []
coef = []
contador = 0
for i in range(0, 1):
    # constantes
    a1 = random.randint(0, 10)
    a2 = random.randint(0, 10)
    b1 = random.randint(0, 10)
    b2 = random.randint(0, 10)
    d1 = random.randint(0, 10)
    d2 = random.randint(0, 10)
    n = random.randint(0, 5)

    coef1 = [a1, a2, b1, b2, d1, d2, n]
    coef.append(coef1)

    def f(t, y):

        # asignar a cada ODE a un elemento de vector
        X = y[0]
        Y = y[1]

        # definimos cada ODE
        # dX_dt=a1/(1+Y**n)-d1*X+b1
        # dY_dt=a2/(1+X**n)-d2*Y+b2
        dX_dt = a1/(1+Y**n)-d1*X+b1
        dY_dt = a2/(1+X**n)-d2*Y+b1

        return [dX_dt, dY_dt]

    # Declaramos el vector de tiempo
    t_span = [0, 50]
    times = np.linspace(t_span[0], t_span[1], 200)

    varfci2 = []
    for i_ci in range(0, 10):
        # Definimos las condiciones iniciales
        y0 = np.array([random.randint(0, 10), random.randint(0, 10)])

        # Resolvemos
        # Modificar manualmente el intervalo de tiempo
        sol = solve_ivp(f, t_span, y0, t_eval=times)
        # sol=solve_ivp(f, t_span, y0) #Dejar que la librería elija el mejor intervalo
        tiempo = sol.t
        var1 = sol.y[0]
        var2 = sol.y[1]

        varfci = np.append([var1], [var2], axis=0)
        varfci2.append(varfci)

       

    varf= varfci2
    print('--------------------------------------------')
    print(coef1)
    print(np.shape(varf))
    contador = contador+1

varf = np.array(varf)
coef = np.array(coef)


# In[59]:


varfnone = varf[None, :]
varfnone = np.transpose(varfnone, (0,3,1,2))


# ## Predecimos con los datos nuevos

# In[60]:


new_predictions = new_model.predict(varfnone)
new_predictions = np.round(new_predictions)
new_predictions = np.clip(new_predictions, 0, None)
print(new_predictions)


# ## Graficamos con los coeficientes reales

# In[61]:


t_span = [0, 50]
times = np.linspace(t_span[0], t_span[1], 200)
plt.figure(figsize=(5,5))
plt.plot(times,np.array(varf)[0,0,:], label=" Concentración X")
plt.plot(times,np.array(varf)[0,1,:], label="Concentración Y")
plt.xlabel('Tiempo')
plt.ylabel('Concentración de genes')
plt.title('Simulación Toggle Swich', fontsize=20)
plt.legend()
plt.show()


# In[62]:


np.shape(new_predictions)


# ## Graficamos con los coeficientes predecidos

# In[63]:


a1 = new_predictions[0,0]
a2 = new_predictions[0,1]
b1 = new_predictions[0,2]
b2 = new_predictions[0,3]
d1 = new_predictions[0,4]
d2 = new_predictions[0,5]
n = new_predictions[0,6]
 
def f(t, y):

    # asignar a cada ODE a un elemento de vector
    X = y[0]
    Y = y[1]

    # definimos cada ODE
    dX_dt = a1/(1+Y**n)-d1*X+b1
    dY_dt = a2/(1+X**n)-d2*Y+b1

    return [dX_dt, dY_dt]

# Declaramos el vector de tiempo
t_span = [0, 50]
times = np.linspace(t_span[0], t_span[1], 200)

# Definimos las condiciones iniciales
y0 = np.array([3,2])

# Resolvemos
# Modificar manualmente el intervalo de tiempo
sol = solve_ivp(f, t_span, y0, t_eval=times)
# sol=solve_ivp(f, t_span, y0) #Dejar que la librería elija el mejor intervalo
tiempo = sol.t
var1 = sol.y[0]
var2 = sol.y[1]

# Graficamos
plt.figure(figsize=(5, 5))
plt.plot(tiempo, var1, label=" Concentración X")
plt.plot(tiempo, var2, label="Concentración Y")
plt.xlabel('Tiempo')
plt.ylabel('Concentración de genes')
plt.title('Simulación Toggle Swich', fontsize=20)
plt.legend()
plt.show()




