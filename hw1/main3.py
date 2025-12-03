import cv2
import numpy as np
import glob
import os

def save_image(img, name="q3_output.png"):
    if img is None:
        print("Failed to load image")
        return
    
    # Save the image with a different name to not overwrite original
    output_path = name
    cv2.imwrite(output_path, img)
    print(f"Image saved as: {output_path}")

def segment_colors(hsv, img, out_prefix="q3"):
    # HSV ranges in OpenCV (H:0-179, S:0-255, V:0-255)
    color_ranges = {
        "red": [ (np.array([0, 70, 50]),  np.array([10, 255, 255])),
                 (np.array([170, 70, 50]), np.array([179, 255, 255])) ],
        "green": [ (np.array([36, 50, 50]), np.array([86, 255, 255])) ],
        "blue": [ (np.array([100, 150, 0]), np.array([140, 255, 255])) ],
        "yellow": [ (np.array([15, 100, 100]), np.array([35, 255, 255])) ]
    }

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    for name, ranges in color_ranges.items():
        mask = None
        for (low, high) in ranges:
            m = cv2.inRange(hsv, low, high)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        # Clean mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        # Apply mask to original image
        segmented = cv2.bitwise_and(img, img, mask=mask)

        # Save mask and segmented result
        save_image(mask, f"{out_prefix}_{name}_mask.png")
        save_image(segmented, f"{out_prefix}_{name}_segmented.png")

if __name__ == '__main__':
    path = "q3.png"
    img = cv2.imread(path)
    if img is None:
        print(f"Failed to read image: {path}")
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        save_image(hsv, "q3_hsv.png")

        # Optionally save separate H, S, V channels (each is single-channel grayscale)
        h, s, v = cv2.split(hsv)
        save_image(h, "q3_h_channel.png")
        save_image(s, "q3_s_channel.png")
        save_image(v, "q3_v_channel.png")

        # Segment common colors and save masks + segmented images
        segment_colors(hsv, img, out_prefix="q3")

