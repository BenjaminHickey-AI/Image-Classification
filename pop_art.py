from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt

def pop_art_halftone(image_path, dot_size=6):
    try:
        # Load and resize image
        img = Image.open(image_path).resize((256, 256))
        img = ImageOps.posterize(img, bits=3)  # Reduce color depth for a pop-art effect

        # Convert to grayscale and enhance contrast
        gray = img.convert('L')
        contrast = ImageEnhance.Contrast(gray).enhance(2.0)

        # Convert grayscale image to NumPy array
        gray_array = np.array(contrast)

        # Create dot-pattern halftone
        halftone = Image.new("RGB", img.size, (255, 255, 255))
        draw = halftone.load()
        for y in range(0, img.height, dot_size):
            for x in range(0, img.width, dot_size):
                # Get brightness (0-255)
                brightness = gray_array[y, x]
                radius = int((255 - brightness) / 255 * (dot_size / 2))

                # Draw dot by filling circle area with black
                for j in range(-radius, radius):
                    for i in range(-radius, radius):
                        if i**2 + j**2 <= radius**2:
                            dx, dy = x + i, y + j
                            if 0 <= dx < img.width and 0 <= dy < img.height:
                                draw[dx, dy] = (0, 0, 0)

        # Blend original color with halftone dots
        blended = Image.blend(img, halftone, alpha=0.5)

        # Display result
        plt.imshow(blended)
        plt.axis('off')
        plt.title("Pop Art Halftone Filter")
        plt.savefig("pop_art_halftone.png")
        print("Saved 'pop_art_halftone.png' with pop-art halftone effect.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    image_path = "basic_cat.jpg"  # Replace with your image path
    pop_art_halftone(image_path)
