

# In[1]:


import tensorflow_datasets as tfds
[test_set_raw, valid_set_raw, train_set_raw], info = tfds.load(
    "tf_flowers",
    split=["train[:10%]", "train[10%:25%]", "train[25%:]"], 
    as_supervised=True,
    with_info=True)


# In[2]:


info


# In[3]:


class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples


# In[4]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
index = 0
sample_images = train_set_raw.take(9) 
for image, label in sample_images:
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title("Class: {}".format(class_names[label])) 
    plt.axis("off")
plt.show(block=False)


# In[5]:


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    return resized_image, label


# In[6]:


import tensorflow as tf
batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# In[7]:


plt.figure(figsize=(8, 8)) 
sample_batch = train_set.take(1)
for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1) 
        plt.imshow(X_batch[index]/255.0)
        plt.title("Class: {}".format(class_names[y_batch[index]])) 
        plt.axis("off")
plt.show()


# In[8]:


from tensorflow.keras import layers, models
from functools import partial
import keras

model = models.Sequential([
    tf.keras.layers.Rescaling(scale=1./125.5, offset=-1),
    keras.layers.Conv2D(filters=32, kernel_size=7, strides=1,
                           padding="same", activation="relu"),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                           padding="same", activation="relu"),
    keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                           padding="same", activation="relu"),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=10, activation='softmax'),
])


# In[9]:


from tensorflow import keras
model.compile(optimizer = "adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[10]:


history = model.fit(train_set, epochs=10, validation_data=valid_set)


# In[11]:


loss_test, acc_test = model.evaluate(test_set)
acc_test


# In[14]:


data = [history.history['accuracy'][-1], history.history['val_accuracy'][-1], acc_test]
data


# In[13]:


import pickle

with open('simple_cnn_acc.pkl', 'wb') as f:
    pickle.dump(data, f)


# In[19]:


import tensorflow.keras.applications.xception

def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image) 
    return final_image, label


# In[20]:


import tensorflow_datasets as tfds
[test_set_raw, valid_set_raw, train_set_raw], info = tfds.load(
    "tf_flowers",
    split=["train[:10%]", "train[10%:25%]", "train[25%:]"], 
    as_supervised=True,
    with_info=True)


# In[21]:


batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# In[22]:


plt.figure(figsize=(8, 8)) 
sample_batch = train_set.take(1)
for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(X_batch[index] / 2 + 0.5)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")
plt.show()


# In[23]:


base_model = tf.keras.applications.xception.Xception( weights="imagenet", include_top=False)


# In[24]:


for index, layer in enumerate(base_model.layers): 
    print(index, layer.name)


# In[25]:


avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)


# In[26]:


for layer in base_model.layers: 
    layer.trainable = False


# In[27]:


optimizer = keras.optimizers.SGD(learning_rate=0.2, momentum=0.9,
                                 decay=0.01)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set,
                    epochs=5)


# In[33]:


for layer in base_model.layers: 
    layer.trainable = True
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9,
                                 nesterov=True, decay=0.001)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_set,
                    validation_data=valid_set,
                    epochs=7)


# In[34]:


loss_test, acc_test = model.evaluate(test_set)
acc_test


# In[37]:


data = [history.history['accuracy'][-1], history.history['val_accuracy'][-1], acc_test]
data


# In[38]:


import pickle

with open('xception_acc.pkl', 'wb') as f:
    pickle.dump(data, f)

