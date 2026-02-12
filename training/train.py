import numpy as np
import tifffile as tiff
from patchify import patchify
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

# Import your model
from models.unet import build_unet


# -------------------------
# Load TIFF images
# -------------------------
print("Loading data...")
large_image_stack = tiff.imread("train_data.tif")
large_mask_stack = tiff.imread("train_label.tif")


# -------------------------
# Patchify and preprocess
# -------------------------
print("Creating patches...")

all_img_patches = []

for img in range(large_image_stack.shape[0]):
    large_img = large_image_stack[img]
    patches_img = patchify(large_img, (256, 256), step=256)

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :]
            single_patch_img = single_patch_img.astype("float32") / 255.0
            all_img_patches.append(single_patch_img)

images = np.array(all_img_patches)
images = np.expand_dims(images, -1)


all_mask_patches = []

for img in range(large_mask_stack.shape[0]):
    large_mask = large_mask_stack[img]
    patches_mask = patchify(large_mask, (256, 256), step=256)

    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            single_patch_mask = patches_mask[i, j, :, :]
            single_patch_mask = single_patch_mask / 255
            all_mask_patches.append(single_patch_mask)

masks = np.array(all_mask_patches)
masks = np.expand_dims(masks, -1)


# -------------------------
# Train/test split
# -------------------------
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    images, masks, test_size=0.25, random_state=0
)


# -------------------------
# Build model
# -------------------------
IMG_HEIGHT = images.shape[1]
IMG_WIDTH = images.shape[2]
IMG_CHANNELS = images.shape[3]

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

print("Building model...")
model = build_unet(input_shape)
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


# -------------------------
# Train
# -------------------------
print("Training...")
history = model.fit(
    X_train,
    y_train,
    batch_size=8,
    epochs=25,
    validation_data=(X_test, y_test),
)


# -------------------------
# Save model
# -------------------------
print("Saving model...")
model.save("unet_model.h5")

print("Training complete.")

