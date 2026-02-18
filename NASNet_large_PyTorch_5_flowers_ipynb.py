# =======================
# STEP 1: Imports
# =======================
import tensorflow as tf
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint  # ✅ Fixed import


# =======================
# STEP 2: Initialize Weights & Biases
# =======================
wandb.init(project="NASNetLarge-flowers", config={
    "epochs": 50,
    "fine_tune_epochs": 20,
    "batch_size": 16,
    "learning_rate": 0.001,
    "fine_tune_lr": 1e-5,
    "architecture": "NASNetLarge",
    "pretrained": True,
    "input_size": 331
})
config = wandb.config


# =======================
# STEP 3: Data Preparation
# =======================
IMAGE_SIZE = (331, 331)

train_dir = "/home/nursena/Downloads/FLowers/flowers/train"
val_dir   = "/home/nursena/Downloads/FLowers/flowers/val"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,     # ✅ Added more augmentation
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=config.batch_size,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=config.batch_size,
    class_mode='categorical'
)

# ✅ Compute steps explicitly to avoid generator issues
steps_per_epoch  = train_generator.samples // config.batch_size
validation_steps = val_generator.samples  // config.batch_size


# =======================
# STEP 4: Load NASNetLarge Model
# =======================
base_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(331, 331, 3))
base_model.trainable = False  # ✅ Cleaner way to freeze all layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)           # ✅ Added dropout to reduce overfitting
predictions = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)


# =======================
# STEP 5: Phase 1 — Train the Head Only
# =======================
model.compile(
    optimizer=Adam(learning_rate=config.learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    WandbMetricsLogger(log_freq="epoch"),
    WandbModelCheckpoint(filepath="model_best.keras", monitor="val_accuracy", save_best_only=True),
    EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True),  # ✅ Added
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)        # ✅ Added
]

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,       # ✅ Fixed
    validation_data=val_generator,
    validation_steps=validation_steps,     # ✅ Fixed
    epochs=config.epochs,
    callbacks=callbacks
)


# =======================
# STEP 6: Phase 2 — Fine-tune the Whole Model  ✅ New
# =======================
base_model.trainable = True  # Unfreeze all layers

model.compile(
    optimizer=Adam(learning_rate=config.fine_tune_lr),  # Much lower LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=config.fine_tune_epochs,
    callbacks=callbacks
)


# =======================
# STEP 7: Finish WandB Run  ✅ New
# =======================
wandb.finish()