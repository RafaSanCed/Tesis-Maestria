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


# In[3]:


tf.test.is_gpu_available()


# # Importamos los datos

# In[1]:


path = '../../../Base de datos/Trayectorias 100 eval'


# ## Datos de entrenamiento

# ### Datos

# In[5]:


varftrain = np.load(path + '/varftrain.npy')


# In[6]:


np.shape(varftrain)


# ### Coeficientes

# In[7]:


coefvarftrain = np.load(path + '/coefvarftrain.npy')


# In[8]:


np.shape(coefvarftrain)


# ## Datos de prueba

# ### Datos

# In[9]:


varftest = np.load(path + '/varftest.npy')


# In[10]:


np.shape(varftest)


# ### Coeficientes

# In[11]:


coefvarftest = np.load(path + '/coefvarftest.npy')


# In[12]:


np.shape(coefvarftest)


# ## Datos de validación

# ### Datos

# In[13]:


varfval = np.load(path + '/varfval.npy')


# In[14]:


np.shape(varfval)


# ### Coeficienes

# In[15]:


coefvarfval = np.load(path + '/coefvarfval.npy')


# In[16]:


np.shape(coefvarfval)


# # Unimos coeficientes con los datos

# In[17]:


train_dataset = tf.data.Dataset.from_tensor_slices((varftrain, coefvarftrain))
test_dataset = tf.data.Dataset.from_tensor_slices((varftest, coefvarftest))
val_dataset = tf.data.Dataset.from_tensor_slices((varfval, coefvarfval))


# In[18]:


train_dataset


# # Mezclar y procesar por lotes los conjuntos de datos

# In[19]:


BATCH_SIZE = 100
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)


# # Creación del modelo

# In[31]:


#model = Sequential()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(10,2,100)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(7)
])


# In[32]:


model.summary()


# # Entrenamos el modelo

# In[33]:


model.compile(optimizer='adam',
              loss='MSE',
              metrics=['mae'])


# In[34]:


def scheduler(epoch, lr):
  if epoch < 100:
    return lr
  else:
    return 1e-3 * 0.99 ** epoch


# In[35]:


val_epochs = 507
tf.compat.v1.global_variables_initializer()

early_stop = tf.keras.callbacks.EarlyStopping( monitor = 'val_loss', patience = 500,verbose = 1, 
                                              restore_best_weights = True)

reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(train_dataset, validation_data=val_dataset, epochs= val_epochs, callbacks=[reduce_lr, early_stop])


# # Analizamos accuracy y loss

# In[37]:


acc = history.history['mae']
val_acc = history.history['val_mae']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(507)

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


# # Guardamos los datos Accuracy y Loss

# In[38]:


df = pd.DataFrame.from_dict(history.history)
df.to_csv('../../../Gráficas finales/historytrayec100.csv', index=False)


# # Guardamos el modelo

# In[1]:


path_to_save = '../../../Modelos/Modelos 50,100'


# In[40]:


model.save(path_to_save + '/trayectorias100.h5')


# # Importamos el modelo

# In[41]:


new_model = keras.models.load_model('../../../Modelos/Modelos 50,100/trayectorias100.h5')


# # Probamos el modelo con datos nuevos

# ## Creamos nuevos datos

# In[42]:


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
    times = np.linspace(t_span[0], t_span[1], 100)

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
    print(contador)

varf = np.array(varf)
coef = np.array(coef)


# In[43]:


varfnone = varf[None, :]


# ## Predecimos con los datos nuevos

# In[44]:


new_predictions = new_model.predict(campvecnone)
new_predictions = np.round(new_predictions)
new_predictions = np.clip(new_predictions, 0, None)
print(new_predictions)


# ## Graficamos con los coeficientes reales

# In[45]:


t_span = [0, 100]
times = np.linspace(t_span[0], t_span[1], 100)
plt.figure(figsize=(5,5))
plt.plot(times,np.array(varf)[0,0,:], label=" Concentración X")
plt.plot(times,np.array(varf)[0,1,:], label="Concentración Y")
plt.xlabel('Tiempo')
plt.ylabel('Concentración de genes')
plt.title('Simulación Toggle Swich', fontsize=20)
plt.legend()
plt.show()


# In[46]:


np.shape(new_predictions)


# ## Graficamos con los coeficientes predecidos

# In[47]:


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
times = np.linspace(t_span[0], t_span[1], 100)

# Definimos las condiciones iniciales
y0 = np.array([10,10])

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


# In[ ]:




