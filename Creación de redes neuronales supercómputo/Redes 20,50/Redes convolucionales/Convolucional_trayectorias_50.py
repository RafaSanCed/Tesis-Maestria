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

# In[4]:


path = '../../../Base de datos/Trayectorias 50 eval'


# ## Datos de entrenamiento

# ### Datos

# In[5]:


varftrain = np.load(path + '/varftrain.npy')
varftrain = np.transpose(varftrain, (0,3,1,2))


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
varftest = np.transpose(varftest, (0,3,1,2))


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
varfval = np.transpose(varfval, (0,3,1,2))


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


BATCH_SIZE = 10
SHUFFLE_BUFFER_SIZE = 10

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)


# # Creación del modelo

# In[20]:


def custom_activation(x):
    return 10 * tf.nn.sigmoid(x)


# In[28]:


model = Sequential()
# Primera capa convolucional
model.add(Conv2D(32, (3, 3), activation='tanh', input_shape=(50, 10, 2),kernel_initializer=HeNormal(), kernel_regularizer=l1(0.001)))
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



# In[29]:


model.summary()


# In[30]:


keras.utils.plot_model(model,show_shapes=True)


# # Entrenamos el modelo

# In[31]:


from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001,clipvalue=10.0)


# In[32]:


model.compile(optimizer= optimizer,
              loss= 'mse',
              metrics=['accuracy'])


# In[33]:


def scheduler(epoch, lr):
  if epoch < 100:
    return lr
  else:
    return 1e-3 -  2e-8* epoch


# In[34]:


val_epochs = 20

early_stop = tf.keras.callbacks.EarlyStopping( monitor = 'val_loss', patience = 800,verbose = 1, 
                                              restore_best_weights = True)

reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(train_dataset, validation_data=val_dataset, epochs= val_epochs, callbacks=[reduce_lr, early_stop])


# # Analizamos accuracy y loss

# In[35]:


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

# In[36]:


df = pd.DataFrame.from_dict(history.history)
df.to_csv('../../../Gráficas finales/History 20,50/historyconvtrayec.csv', index=False)


# # Guardamos el modelo

# In[37]:


path_to_save = '../../../Modelos/Modelos 20,50/'


# In[38]:


model.save(path_to_save + '/convolucionaltr.h5')


# # Importamos el modelo

# In[39]:


custom_objects = {'custom_activation': custom_activation}
new_model = keras.models.load_model('../../../Modelos/Modelos 20,50/convolucionaltr.h5', custom_objects=custom_objects)


# # Probamos el modelo con datos nuevos

# ## Creamos nuevos datos

# In[46]:


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
    times = np.linspace(t_span[0], t_span[1], 50)

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


# In[47]:


varfnone = varf[None, :]
varfnone = np.transpose(varfnone, (0,3,1,2))


# ## Predecimos con los datos nuevos

# In[48]:


new_predictions = new_model.predict(varfnone)
new_predictions = np.round(new_predictions)
new_predictions = np.clip(new_predictions, 0, None)
print(new_predictions)


# ## Graficamos con los coeficientes reales

# In[50]:


t_span = [0, 50]
times = np.linspace(t_span[0], t_span[1], 50)
plt.figure(figsize=(5,5))
plt.plot(times,np.array(varf)[0,0,:], label=" Concentración X")
plt.plot(times,np.array(varf)[0,1,:], label="Concentración Y")
plt.xlabel('Tiempo')
plt.ylabel('Concentración de genes')
plt.title('Simulación Toggle Swich', fontsize=20)
plt.legend()
plt.show()


# In[51]:


np.shape(new_predictions)


# ## Graficamos con los coeficientes predecidos

# In[52]:


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
times = np.linspace(t_span[0], t_span[1], 50)

# Definimos las condiciones iniciales
y0 = np.array([10,5])

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


# # Pruebas mensas

# In[ ]:


new_predictions = np.round(new_predictions)
new_predictions = np.clip(new_predictions, 0, None)


# In[ ]:





# In[70]:


def custom_loss2(y_true2, y_pred2):
    campvec = []
    # constantes
    a1 = y_true2[0]

    a2 =y_true2[1]
    
    b1 = y_true2[2]
    
    b2 = y_true2[3]

    d1 = y_true2[4]

    d2 = y_true2[5]

    n = y_true2[5]
    def f(t, y):

        # asignar a cada ODE a un elemento de vector
        X = y[0]
        Y = y[1]

        # definimos cada ODE
        dX_dt = a1/(1+Y**n)-d1*X+b1
        dY_dt = a2/(1+X**n)-d2*Y+b1

        return [dX_dt, dY_dt]

    x_range_neg = 0
    x_range_pos = 5
    y_range_neg = 0
    y_range_pos =5

    y1 = np.linspace(x_range_neg, x_range_pos, 100)
    y2 = np.linspace(y_range_neg, y_range_pos, 100)

    Y1, Y2 = np.meshgrid(y1, y2)
    
    t1 = 0

    u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)

    NI, NJ = Y1.shape

    u1, v1 = f(t1, [Y1, Y2])
    u, v = f(t1, [Y1, Y2])
    M = (np.hypot(u, v))
    u /= M
    v /= M
# ----------------------------------------------------------------------------
    campvec2 = []
    # constantes
    a12 = y_pred2[0]

    a22 =y_pred2[1]
    
    b12 = y_pred2[2]
    
    b22 = y_pred2[3]

    d12 = y_pred2[4]

    d22 = y_pred2[5]

    n2 = y_pred2[5]
    def f2(t2, ye2):

        # asignar a cada ODE a un elemento de vector
        X2 = ye2[0]
        Y2 = ye2[1]

        # definimos cada ODE
        dX2_dt = a12/(1+Y2**n2)-d12*X2+b12
        dY2_dt = a22/(1+X2**n2)-d22*Y2+b12
        return [dX2_dt, dY2_dt]

    y12 = np.linspace(x_range_neg, x_range_pos, 100)
    y22 = np.linspace(y_range_neg, y_range_pos, 100)
    Y12, Y22 = np.meshgrid(y12, y22)
    u2, v2 = np.zeros(Y12.shape), np.zeros(Y22.shape)
    u2, v2 = f2(t1, [Y12, Y22])
    M2 = np.sqrt(np.square(u2) + np.square(v2))
    u2 /= M2
    v2 /= M2
    campvecx = u- u2
    campvecy = v - v2    
    
    
    #loss = tf.reduce_mean(campvectotal)
    return u,v, u2,v2, campvecx, campvecy,Y1,Y2,Y12,Y22,M,M2


# In[71]:


u,v,u2,v2,campvecx,campvecy,Y1,Y2,Y12,Y22,M,M2 = custom_loss2(y_true2,y_pred2)
print(u)


# In[72]:


x_range_neg = 0
x_range_pos = 5
y_range_neg = 0
y_range_pos =5
plt.figure(figsize=(10, 10))
Q = plt.quiver(Y1, Y2, u, v, M, angles='xy')
plt.title('Campo vectorial espacio fase Toggle Swich', fontsize=20)
plt.xlim([x_range_neg, x_range_pos])
plt.ylim([y_range_neg, y_range_pos])
plt.xlabel('Concentración X')
plt.ylabel('Concentración Y')


# In[73]:


x_range_neg = 0
x_range_pos = 5
y_range_neg = 0
y_range_pos =5
plt.figure(figsize=(10, 10))
Q = plt.quiver(Y12, Y22, u2, v2, M2, angles='xy')
plt.title('Campo vectorial espacio fase Toggle Swich', fontsize=20)
plt.xlim([x_range_neg, x_range_pos])
plt.ylim([y_range_neg, y_range_pos])
plt.xlabel('Concentración X')
plt.ylabel('Concentración Y')


# In[76]:


x_range_neg = 0
x_range_pos = 5
y_range_neg = 0
y_range_pos =5
plt.figure(figsize=(10, 10))
Q = plt.quiver(Y12, Y22, campvecx, campvecy, angles='xy')
plt.title('Campo vectorial espacio fase Toggle Swich', fontsize=20)
plt.xlim([x_range_neg, x_range_pos])
plt.ylim([y_range_neg, y_range_pos])
plt.xlabel('Concentración X')
plt.ylabel('Concentración Y')


# In[ ]:




