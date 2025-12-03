import cv2
import numpy as np
import glob
import os

def save_image(img, name="q4_output.png"):
    if img is None:
        print("Failed to load image")
        return
    
    # Save the image with a different name to not overwrite original
    output_path = name
    cv2.imwrite(output_path, img)
    print(f"Image saved as: {output_path}")

if __name__ == '__main__':
    path = "q4.png"
    img = cv2.imread(path)
    if img is None:
        print(f"Failed to read image: {path}")
    else:
        for i in img:
            for j in i:
                j[2] = int(-j[2]-j[1]-j[0]) % 256 # Invert Red channel

        save_image(img, "q4_inverted.png")