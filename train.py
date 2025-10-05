import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

DATA_DIR = 'C:/Users/arese/Downloads/butterfly_dataset'
IMG_SIZE = (224, 224) # Standard size for VGG16
BATCH_SIZE = 32

# Create datasets
train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Get class names for later use in prediction
class_names = train_ds.class_names
num_classes = len(class_names)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Load the pre-trained VGG16 base model
base_model = VGG16(
    weights='imagenet',
    include_top=False, # Important: Excludes the ImageNet classification layers
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# 2. Freeze the base layers
# This prevents the pre-trained weights from being updated during training.
base_model.trainable = False
# 3. Create the new Sequential model
model = Sequential([
    # Data Augmentation layer to prevent overfitting (optional but recommended)
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    
    # The frozen VGG16 base model
    base_model,
    
    # Flatten the output of the VGG16 base
    Flatten(),
    
    # Custom layers for classification
    Dense(512, activation='relu'),
    Dropout(0.5), # Regularization layer to prevent overfitting
    
    # Final output layer: 'num_classes' neurons with softmax activation
    Dense(num_classes, activation='softmax')
])
model.compile(
    optimizer=Adam(learning_rate=0.0001), # Small learning rate for fine-tuning
    loss='sparse_categorical_crossentropy', # Appropriate for integer labels
    metrics=['accuracy']
)

model.summary()
EPOCHS = 10 # Start with 10-20 epochs, adjust based on results

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
# Save the model to the file path your app.py is expecting
MODEL_PATH = 'models/vgg16_model.h5' 
model.save(MODEL_PATH)
print(f"Model saved successfully as {MODEL_PATH}")