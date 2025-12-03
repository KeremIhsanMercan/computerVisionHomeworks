import pyautogui 
import time 
import cv2
import numpy as np
import threading
 
def extract_between(screenshot, x1, y1, x2, y2):
    width, height = screenshot.size
    left = int(x1 * width)
    top = int(y1 * height)
    right = int(x2 * width)
    bottom = int(y2 * height)
    return screenshot.crop((left, top, right, bottom))

SHAPE_TO_KEY = {
    "Triangle": "a",
    "Square": "b",
    "Hexagon": "f",
    "Star": "d",
}

PRESS_INTERVAL = 0.01

shape_lock = threading.Lock()
shape_changed = threading.Event()
stop_event = threading.Event()
current_shape = "No shape"

def extract_pixel_color(screenshot, x, y):
    width, height = screenshot.size
    pixel_x = int(x * width)
    pixel_y = int(y * height)
    return screenshot.getpixel((pixel_x, pixel_y))

def is_black(color, threshold=10):
    r, g, b = color
    return r < threshold and g < threshold and b < threshold

def detect_black_shape_from_whitebackground_with_minimum_eigenvalue(cropped_image):
    open_cv_image = np.array(cropped_image) 
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    inverted_image = cv2.bitwise_not(gray_image)

    corners = cv2.goodFeaturesToTrack(inverted_image, maxCorners=10, qualityLevel=0.2, minDistance=10)
    if corners is not None:
        if len(corners) == 3:
            return "Triangle"
        elif len(corners) == 4:
            return "Square"
        elif len(corners) == 6:
            return "Hexagon"
        elif len(corners) == 10:
            return "Star"
        else:
            return "No shape"
    else:
        return "No shape"

# In this 2 seconds you should switch to game screen to transfer the 
# simulated keyboard inputs to the game. 
print("GO!")
time.sleep(1) 
def update_shape(new_shape: str) -> None:
    global current_shape
    if new_shape == "No shape":
        return
    with shape_lock:
        if new_shape == current_shape:
            return
        current_shape = new_shape
    shape_changed.set()


def key_press_worker() -> None:
    global current_shape
    while not stop_event.is_set():
        shape_changed.wait(timeout=0.1)
        shape_changed.clear()
        if stop_event.is_set():
            break
        with shape_lock:
            shape = current_shape

        if shape not in SHAPE_TO_KEY:
            continue

        key = SHAPE_TO_KEY[shape]
        # Keep pressing the mapped key until a new shape arrives.
        while not stop_event.is_set():
            with shape_lock:
                if current_shape != shape:
                    break
            pyautogui.press(key)
            time.sleep(PRESS_INTERVAL)


def main() -> None:
    global current_shape
    print("GO!")
    time.sleep(1)

    worker = threading.Thread(target=key_press_worker, daemon=True)
    worker.start()

    try:
        while True:
            myScreenshot = pyautogui.screenshot() 
            myScreenshot.save('ss/test.png')

            print(myScreenshot.size)

            cropped_image = extract_between(myScreenshot, 0, 810/1080, 1, 1)

            cropped_area = extract_between(cropped_image, 0.55, 0, 0.72, 1)
            cropped_area.save('ss/cropped.png')

            shape = detect_black_shape_from_whitebackground_with_minimum_eigenvalue(cropped_area)
            print("Detected shape:", shape, " / Current shape:", current_shape)
            update_shape(shape)

            time.sleep(0.3)
    except KeyboardInterrupt:
        print("Stopping detection loop...")
    finally:
        stop_event.set()
        shape_changed.set()
        worker.join()


if __name__ == "__main__":
    main()
