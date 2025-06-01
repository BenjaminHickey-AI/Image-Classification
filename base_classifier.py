import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load pre-trained MobileNetV2
model = MobileNetV2(weights="imagenet")

# === Grad-CAM ===
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(inputs=[model.inputs],
                       outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_img_path, alpha=0.4):
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed_img

# === Occlusion Methods ===
def black_box_occlusion(img, x, y, size):
    img_occluded = img.copy()
    img_occluded[y:y+size, x:x+size] = 0
    return img_occluded

def blur_occlusion(img, x, y, size):
    img_occluded = img.copy()
    roi = img_occluded[y:y+size, x:x+size]
    roi_blur = cv2.GaussianBlur(roi, (15, 15), 0)
    img_occluded[y:y+size, x:x+size] = roi_blur
    return img_occluded

def noise_patch_occlusion(img, x, y, size):
    img_occluded = img.copy()
    noise = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    img_occluded[y:y+size, x:x+size] = noise
    return img_occluded

# === Prediction & Visualization ===
def classify_image(image_path):
    try:
        # Load and preprocess
        orig = image.load_img(image_path, target_size=(224, 224))
        orig_array = image.img_to_array(orig).astype(np.uint8)
        img_input = preprocess_input(np.expand_dims(orig_array.copy(), axis=0))

        # Predict
        predictions = model.predict(img_input)
        decoded = decode_predictions(predictions, top=3)[0]
        print("Top-3 Predictions:")
        for i, (_, label, score) in enumerate(decoded):
            print(f"{i+1}: {label} ({score:.2f})")

        # Grad-CAM
        heatmap = make_gradcam_heatmap(img_input, model, 'Conv_1')
        cam_img = overlay_heatmap(heatmap, image_path)

        # Coordinates for occlusion (centered)
        x, y, size = 80, 80, 60

        # Apply occlusions
        black_box_img = black_box_occlusion(orig_array, x, y, size)
        blur_img = blur_occlusion(orig_array, x, y, size)
        noise_img = noise_patch_occlusion(orig_array, x, y, size)

        # Display everything
        fig, axes = plt.subplots(1, 5, figsize=(18, 5))
        axes[0].imshow(orig)
        axes[0].set_title("Original")
        axes[1].imshow(cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Grad-CAM")
        axes[2].imshow(black_box_img.astype(np.uint8))
        axes[2].set_title("Black Box")
        axes[3].imshow(blur_img.astype(np.uint8))
        axes[3].set_title("Blurred")
        axes[4].imshow(noise_img.astype(np.uint8))
        axes[4].set_title("Noise Patch")
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

# Main
if __name__ == "__main__":
    image_path = "basic_cat.jpg"  # Change to your own image
    classify_image(image_path)
