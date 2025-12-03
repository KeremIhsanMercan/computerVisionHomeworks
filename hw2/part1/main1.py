import moviepy.video.io.VideoFileClip as mpy 
import cv2 
import numpy as np
 
vid = mpy.VideoFileClip('shapes_video.mp4') 
 
framecount = vid.reader.n_frames 
videofps = vid.fps 
 
def save_frame_as_image(frame, index, origin=False):
    
    if origin:
        filename = f"frames/frame_{index:04d}_origin_.png"
    else:
        filename = f"frames/frame_{index:04d}.png" 
    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

def compare_frames_and_get_newly_occurred_card_and_paint_everywhere_else_red(
    current_frame,
    previous_frame,
    diff_threshold=25,
    min_area_ratio=0.01,
):
    """Return only the newly appeared card between two consecutive frames."""
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
    prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_RGB2GRAY)

    frame_diff = cv2.absdiff(curr_gray, prev_gray)
    _, diff_mask = cv2.threshold(frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)
    diff_mask = cv2.medianBlur(diff_mask, 5)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    edges = cv2.Canny(curr_gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    new_edges = cv2.bitwise_and(edges, diff_mask)

    contours, _ = cv2.findContours(new_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = curr_gray.shape[0] * curr_gray.shape[1]
    min_area = max(int(image_area * min_area_ratio), 1500)

    card_contour = None
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) >= 4:  # allow rotated rectangles
            card_contour = approx
            break

    if card_contour is None:
        fallback_contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not fallback_contours:
            return None
        fallback = max(fallback_contours, key=cv2.contourArea)
        if cv2.contourArea(fallback) < min_area:
            return None
        card_contour = fallback
    mask = np.zeros_like(curr_gray)
    cv2.drawContours(mask, [card_contour], -1, 255, thickness=cv2.FILLED)
    new_card = cv2.bitwise_and(current_frame, current_frame, mask=mask)
    red_background = np.full_like(current_frame, (255, 0, 0))
    red_background = cv2.bitwise_and(red_background, red_background, mask=cv2.bitwise_not(mask))
    result = cv2.add(new_card, red_background)
    return result

shape_counts = {}
shape_counts["Triangle"] = 0
shape_counts["Square"] = 0
shape_counts["Pentagon"] = 0
shape_counts["Hexagon"] = 0
shape_counts["Sevengon"] = 0
shape_counts["Octagon"] = 0
shape_counts["Nonagon"] = 0
shape_counts["Star"] = 0
shape_counts["Circle"] = 0

def detect_black_shape_from_white_card_on_red_background(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower_black, upper_black)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)

    shape_name = None
    vertex_count = len(approx)

    if vertex_count == 3:
        shape_name = "Triangle"
    elif vertex_count == 4:
        shape_name = "Square"
    elif vertex_count == 5 or vertex_count == 6:
        shape_name = "Pentagon"
    elif vertex_count == 7:
        shape_name = "Sevengon"
    elif vertex_count == 8:
        shape_name = "Octagon"
    elif vertex_count == 9:
        shape_name = "Nonagon"
    elif vertex_count >= 10:
        shape_name = "Star"
    else:
        shape_name = "Circle"
    
    shape_counts[shape_name] += 1
    return shape_name



for i in range(framecount): 
    if i < 6:
        continue
    if i > 81:
        break
    frame = vid.get_frame(i * 1.0 / videofps)

    frame = cv2.medianBlur(frame, 3)
    save_frame_as_image(frame, i, origin=True)

    if i > 0:
        previous_frame = vid.get_frame((i - 1) * 1.0 / videofps)
        previous_frame = cv2.medianBlur(previous_frame, 3)
        frame = compare_frames_and_get_newly_occurred_card_and_paint_everywhere_else_red(frame, previous_frame)
    else:
        frame = None

    if frame is not None:
        save_frame_as_image(frame, i)
        print("Detected shape in frame {}: {}".format(i, detect_black_shape_from_white_card_on_red_background(frame) if frame is not None else "No new card"))
    
print("Squares detected:", shape_counts["Square"])
print("Pentagons detected:", shape_counts["Pentagon"])
print("Stars detected:", shape_counts["Star"])