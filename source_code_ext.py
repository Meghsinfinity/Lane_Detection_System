import cv2
import numpy as np
from collections import deque

# For smoothing lines over frames
left_line_history = deque(maxlen=10)
right_line_history = deque(maxlen=10)

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])  # bottom of the image
    y2 = int(y1 * 3 / 5)      # slightly lower than the middle
    if slope == 0: slope = 0.1  # Avoid division by zero
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            try:
                fit = np.polyfit((x1, x2), (y1, y2), 1)  # y = mx + c
                slope, intercept = fit
                if slope < 0:  # Negative slope -> left lane
                    left_fit.append((slope, intercept))
                else:  # Positive slope -> right lane
                    right_fit.append((slope, intercept))
            except np.RankWarning:
                continue

    # Apply moving average smoothing
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line_history.append(left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line_history.append(right_fit_average)

    left_fit_average = np.average(left_line_history, axis=0) if left_line_history else None
    right_fit_average = np.average(right_line_history, axis=0) if right_line_history else None

    left_line = make_points(image, left_fit_average) if left_fit_average is not None else None
    right_line = make_points(image, right_fit_average) if right_fit_average is not None else None

    return [left_line, right_line]

def canny_edge_detection(img, low_threshold=50, high_threshold=150):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, low_threshold, high_threshold)

def region_of_interest(image):
    height, width = image.shape[:2]
    mask = np.zeros_like(image)
    polygon = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.45), int(height * 0.6)),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.9), height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image, mask)

def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line[0]
                slope = abs((y2 - y1) / (x2 - x1)) if x2 != x1 else 1
                color = (0, int(255 - slope * 100), 255)  # Color changes based on slope
                cv2.line(line_image, (x1, y1), (x2, y2), color, 10)
    return line_image

def highlight_lane_area(image, lines):
    lane_image = np.zeros_like(image)
    if lines[0] is not None and lines[1] is not None:
        left_line = lines[0][0]
        right_line = lines[1][0]
        points = np.array([[left_line[0], left_line[1]], [left_line[2], left_line[3]],
                           [right_line[2], right_line[3]], [right_line[0], right_line[1]]], np.int32)
        cv2.fillPoly(lane_image, [points], (0, 255, 0))
    return cv2.addWeighted(image, 1, lane_image, 0.3, 0)

def process_video(input_video, output_video="output.avi"):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        canny_image = canny_edge_detection(frame)
        cropped_canny = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = average_slope_intercept(frame, lines)
        lane_lines = display_lines(frame, averaged_lines)
        lane_highlight = highlight_lane_area(frame, averaged_lines)
        final_output = cv2.addWeighted(lane_highlight, 1, lane_lines, 1, 0)

        cv2.imshow("Lane Detection", final_output)
        out.write(final_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

process_video("test3.mp4")