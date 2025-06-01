from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

def apply_multiple_filters(image_path):
    try:
        # Open and resize the image
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))

        # Apply different filters
        sharpened = img_resized.filter(ImageFilter.SHARPEN)
        edge_enhanced = img_resized.filter(ImageFilter.EDGE_ENHANCE)
        edge_detected = img_resized.filter(ImageFilter.FIND_EDGES)

        # Plot the original and filtered images
        fig, axes = plt.subplots(1, 4, figsize=(12, 4))

        axes[0].imshow(img_resized)
        axes[0].set_title("Original")
        axes[1].imshow(sharpened)
        axes[1].set_title("Sharpen")
        axes[2].imshow(edge_enhanced)
        axes[2].set_title("Edge Enhance")
        axes[3].imshow(edge_detected)
        axes[3].set_title("Find Edges")

        # Hide axis ticks
        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig("filtered_image_comparison.png")
        print("Filtered image comparison saved as 'filtered_image_comparison.png'.")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    image_path = "basic_cat.jpg"  # Replace with your image path
    apply_multiple_filters(image_path)
