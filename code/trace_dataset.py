import os 
import pandas as pd 
import numpy as np
import cv2
import time

np.set_printoptions(suppress=True)

# Define colors
colors = {
    'red': (0, 0, 255),
    'orange': (0, 165, 255),
    'green': (0, 255, 0)
}

def calculate_threshold(data):
    x = (data[:, 0] > 499) & (data[:, 0] < 501)
    final = data[x]
    return np.mean(final[:, 1])

def track_and_trace_red_color_band(frame, trace, color,thr):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([150, 45, 215])
    upper_red = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m=len(trace)
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 10, (255, 192, 203), -1)
            cord=(cx, cy)
            trace.append([cord,thr])

    for i in range(1, len(trace)):
        th=trace[i][1]
        if th > 0.7:
            color = colors['red']
        elif th > 0.5:
            color = colors['orange']
        else:
            color = colors['green']
        cv2.line(frame, trace[i - 1][0], trace[i][0], color, 2)

    return frame

files_dir = r'Created dataset_Probe tracking\Dataset'
files = os.listdir(files_dir)

cap = cv2.VideoCapture(0)

color_index = 0
c=0
trace=[]
while True:
    color_name = list(colors.keys())[color_index]
    color = colors[color_name]
    if c==1:
        break
    for file_name in files:
        if c==1:
            break
        path = os.path.join(files_dir, file_name)
        data = pd.read_csv(path)
        data = np.array(data[7:], dtype=float)

        thr = calculate_threshold(data)
        print("current file is ",file_name)
        print("Threshold :",thr)
        print('')
        if thr > 0.7:
            color = colors['red']
        elif thr > 0.5:
            color = colors['orange']
        else:
            color = colors['green']

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            result_frame = track_and_trace_red_color_band(frame, trace, color,thr)
            cv2.imshow('Red Color Band Tracking with Color Change', result_frame)
            key=cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                c=1
                break
            if key == ord('q') or time.time() - start_time >= 2:
                break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    
    color_index = (color_index + 1) % len(colors)  # Change the color index for the next iteration

cap.release()
cv2.destroyAllWindows()