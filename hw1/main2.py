import cv2
import numpy as np
import glob
import os

def open_q2_image(img, name="q2_output.png"):
    if img is None:
        print("Failed to load image")
        return
    
    # Save the image with a different name to not overwrite original
    output_path = name
    cv2.imwrite(output_path, img)
    print(f"Image saved as: {output_path}")

def plot_gamma_curves(gammas, out_path="gamma_curves.png"):
    """
    Compute and save a plot of the transformation curve for each gamma.
    x axis: input intensity [0..255]
    y axis: output intensity [0..255] after gamma correction
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib is required to plot curves. Install it (pip install matplotlib).")
        return

    x = np.arange(256, dtype=np.float32) / 255.0
    plt.figure()
    for g in gammas:
        y = np.power(x, g)
        y255 = (y * 255.0).clip(0, 255)
        plt.plot(np.arange(256), y255, label=f"gamma={g}")
    plt.title("Gamma transformation curves")
    plt.xlabel("Input intensity (0-255)")
    plt.ylabel("Output intensity (0-255)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Gamma curves saved as: {out_path}")
    plt.show()

if __name__ == '__main__':
    path = "q2.png"
    img = cv2.imread(path)
    if img is None:
        print(f"Failed to read image: {path}")
    else:
        gammas = [0.4, 1.0, 2.2]

        img1 = np.power(img/255.0, gammas[0]) * 255.0
        img1 = img1.astype(np.uint8)
        open_q2_image(img1, "q2_output_0.4.png")

        img2 = np.power(img/255.0, gammas[1]) * 255.0
        img2 = img2.astype(np.uint8)
        open_q2_image(img2, "q2_output_1.0.png")

        img3 = np.power(img/255.0, gammas[2]) * 255.0
        img3 = img3.astype(np.uint8)
        open_q2_image(img3, "q2_output_2.2.png")

        # extract and display/save the transformation curves
        plot_gamma_curves(gammas)