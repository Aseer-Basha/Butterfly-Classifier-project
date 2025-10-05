import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Specify the path to your image folder
folder_path = r'C:/Users/arese/Downloads/butterfly_dataset/train/Nymphalidae' 

# List all files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Select a random image from the list
selected_image = random.choice(image_files)

# Get the full path to the image
image_path = os.path.join(folder_path, selected_image)

# Read and display the image using Matplotlib
img = mpimg.imread(image_path)
plt.imshow(img)
plt.show()

import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Specify the path to your image folder
folder_path = r'C:/Users/arese/Downloads/butterfly_dataset/test/Nymphalidae' 

# List all files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Select a random image from the list
selected_image = random.choice(image_files)

# Get the full path to the image
image_path = os.path.join(folder_path, selected_image)

# Read and display the image using Matplotlib
img = mpimg.imread(image_path)
plt.imshow(img)
plt.show()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
trainpath='C:/Users/arese/Downloads/butterfly_dataset/train'

testpath='C:/Users/arese/Downloads/butterfly_dataset/test'

train_datagen = ImageDataGenerator(rescale = 1./255,zoom_range= 0.2, shear_range= 8.2)

test_datagen = ImageDataGenerator(rescale = 1./255)

train = train_datagen.flow_from_directory(trainpath, target_size =(224, 224), batch_size = 20)

test = test_datagen.flow_from_directory(testpath, target_size = (224, 224), batch_size = 20),#5,15, 32, 50

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import numpy as np # Used here to simulate a model input for testing 'summary()'
vgg = VGG16(include_top=False, input_shape=(224, 224, 3))

print("\n--- VGG16 Base Layers ---")
for layer in vgg.layers:
    print(layer)

# 2. Freeze the layers of the base VGG16 model
# This prevents their weights from being updated during new training
for layer in vgg.layers:
    layer.trainable = False

# Print the number of layers in the VGG model (Should be 19)
print(f"\nNumber of layers in vgg.layers: {len(vgg.layers)}")

# 3. Define the new classification head
# Get the output of the base model
x = vgg.output

# Flatten the output from the convolutional layers (pooling/feature maps)
x = Flatten()(x)


output = Dense(28, activation='softmax')(x)


vgg16 = Model(inputs=vgg.input, outputs=output)


print("\n--- VGG16 Transfer Learning Model Summary ---")
vgg16.summary()



import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# 1. Load the pre-trained VGG16 model
# Include weights='imagenet' to use the model trained on the ImageNet dataset
vgg16 = VGG16(weights='imagenet')

# 2. Define the image path (CHANGE THIS to your actual local path)
# Make sure you have an image file named 'Image 1565.jpg' or change the name.
img_path = 'C:/Users/arese/Downloads/butterfly_dataset/train/Nymphalidae/Image_1565.jpg' 

# 3. Load the image and resize it to VGG16's required input size (224x224)
img = image.load_img(img_path, target_size=(224, 224))

# 4. Convert the image to a NumPy array
x = image.img_to_array(img)

# 5. Add a dimension to make it a batch of 1 image (required by Keras model)
x = np.expand_dims(x, axis=0)

# 6. Apply VGG16's specific preprocessing (scaling, mean subtraction, etc.)
x = preprocess_input(x)

# 7. Get predictions from the VGG16 model
preds = vgg16.predict(x)

print(preds)

# ... continuing from the code in Snippet A ...
from tensorflow.keras.applications.vgg16 import decode_predictions

# Get the top 5 predicted classes and their probabilities
# 'preds' contains the prediction scores for 1000 ImageNet classes
decoded_preds = decode_predictions(preds, top=5)[0] 

# The first element of the list will be the top prediction (class index, name, probability)
predicted_class_name = decoded_preds[0][1] # Get the name of the top class

# Get the index of the highest probability (the predicted class index)
predicted_class_index = np.argmax(preds, axis=1)[0]


print(f"\nPredicted Class Index: {predicted_class_index}")
print(f"Predicted Class Name (Top 1): {predicted_class_name.upper()}")

# Example output for top 5 (optional)
print("\nTop 5 Predictions:")
for i, (imagenet_id, name, prob) in enumerate(decoded_preds):
    print(f"{i+1}: {name} ({prob*100:.2f}%)")


vgg16.save(r'C:\Users\arese\Desktop\Saved_Models\vgg16_model.h5python python python app.pyp[]')