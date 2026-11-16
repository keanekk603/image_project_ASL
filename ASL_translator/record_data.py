import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
from pathlib import Path
import pandas as pd

SESSION_ID = time.strftime("%Y%m%d-%H%M%S")
motion_mode = False
input_variable = None
FRAME_COUNTER = 0
START_RECORDING = False
CLIP = 0
MOTION_LOG_BUFFER = []

# The formal MediaPipe skeletal path mapping connecting all 21 joints together
HAND_CONNECTIONS = [
    # Base Palm / Wrist
    (0, 1), (0, 5), (0, 17),
    # Thumb path
    (1, 2), (2, 3), (3, 4),
    # Index Finger path
    (5, 6), (6, 7), (7, 8),
    # Middle Finger path
    (9, 10), (10, 11), (11, 12),
    # Ring Finger path
    (13, 14), (14, 15), (15, 16),
    # Pinky Finger path
    (17, 18), (18, 19), (19, 20),
    # Knuckle-to-knuckle connections closing the palm loop structure
    (5, 9), (9, 13), (13, 17)
]

# Configure the Tasks API options for VIDEO mode
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

    
cap = cv2.VideoCapture(0) # Change to 1 if using an external or continuity camera

print("Starting Hand Tracker... Press 'esc' in the video window to exit.")
def main():
    write_static_header()
    write_motion_header()
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            frame_timestamp_ms = int(time.time() * 1000)
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if detection_result.hand_landmarks:
                h, w, _ = frame.shape

                for hand_indx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    # 1. First, convert all normalized landmarks into pixel tuples and store them
                    joints = []
                    for landmark in hand_landmarks:
                        pixel_x = int(landmark.x * w)
                        pixel_y = int(landmark.y * h)
                        joints.append((pixel_x, pixel_y))

                    # 2. Draw the structural connecting lines BETWEEN the joints first
                    for start_joint, end_joint in HAND_CONNECTIONS:
                        start_pos = joints[start_joint]
                        end_pos = joints[end_joint]
                        # Draw a red line connecting the two joints (thickness = 2)
                        cv2.line(frame, start_pos, end_pos, (0, 0, 255), 2)

                    # 3. Draw the green joint dots on TOP of the lines so it looks clean
                    for (pixel_x, pixel_y) in joints:
                        cv2.circle(frame, (pixel_x, pixel_y), 5, (0, 255, 0), cv2.FILLED)

            cv2.imshow("Record data", frame)
            key = cv2.waitKey(1) & 0xFF

            #global motion_mode, when i change this i change the global variable
            global motion_mode, input_variable, START_RECORDING, FRAME_COUNTER, CLIP, MOTION_LOG_BUFFER

            #pass the key in
            if detection_result.hand_landmarks:
                write_data(detection_result, key)

            #record motion
            if key == ord('1'):
                #if mode is false set to true
                if motion_mode == False:
                    input_variable = input("Enter: ")
                    motion_mode = True
                    print("motion mode")

                #if mode is true set to false
                else:
                    motion_mode = False 
                    input_variable = None
                    print("exiting motion mode")

            #if start reocrdinf through get record motion and if got hand
            if START_RECORDING == True and detection_result.hand_landmarks:
                motion_record(detection_result)

            #if key is space
            if key == ord(' '):
                if motion_mode == True and input_variable and START_RECORDING == False:
                    if start_timer():
                        START_RECORDING = True
                
                #reset all the stuff
                elif motion_mode == True and input_variable and START_RECORDING == True:
                    CLIP += 1
                    FRAME_COUNTER = 0
                    START_RECORDING = False
                    motion_write_data()
                    MOTION_LOG_BUFFER = []
                    print("recording stop")
            
            if key == 27: # Esc quits
                break

    cap.release()
    cv2.destroyAllWindows()

"""
    result.hand_landmarks = [
    [ point0, point1, ..., point20 ],   # hand 0
    [ point0, point1, ..., point20 ]]    # hand 1
    loop throguh each hand then each point which has an x y z attribute
    result.handedness = [
    [                                                   # hand 0
        Category(category_name="Right", score=0.97),      #   guess [0], best
        Category(category_name="Left",  score=0.03),      #   guess [1]
    ],
    [                                                   # hand 1
        Category(category_name="Left",  score=0.95),      #   guess [0], best
        Category(category_name="Right", score=0.05),      #   guess [1]
    ]] to call for the first hand need the first list and the first guess, thats why we need hand_indx
    """
def motion_write_data():
    #write the data in motion log buffer into the motion file
    with open("data/motion.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        #get the list (which is the values), each list is the length of the toal recording
        for row in MOTION_LOG_BUFFER:
            row_to_write = list(row.values())
            writer.writerow(row_to_write)

def write_data(result, key):

    #ord converts the string to unicode
    if key < ord('z') and key >= ord('a'):
        for hand_indx, hand_landmarks in enumerate(result.hand_landmarks):
            #1. get the letter
            keypressed = chr(key).upper()

            #2. get which hand [result.handedness = [guess1, guess2], [guess1, guess2]], hand_indx, tells which hand we looking out, the 0 tells the first guess
            handness = result.handedness[hand_indx][0].category_name

            #3. get the data
            landmark_data = []
            #loop throguh each landmark object and append it, the x y z
            for landmark in hand_landmarks:
                landmark_data += [landmark.x, landmark.y, landmark.z]

            body = [keypressed, SESSION_ID, handness] + landmark_data
            
            #open('link', 'mode', newline='' (basically to put a \n at the end), encoding='' (the style that is written in))
            with open("data/test_static.csv", 'a', newline="", encoding='utf-8') as f:
                writer = csv.writer(f)
                #write the body data in
                writer.writerow(body)
    

def motion_record(result):
    #increment the counter
    global FRAME_COUNTER, MOTION_LOG_BUFFER
    FRAME_COUNTER += 1

    #iniitliase a row
    row = {"label": input_variable, "session_id": SESSION_ID, "clip":CLIP, "frame": FRAME_COUNTER}

    for hand in ["Right", "Left"]:
        for i in range(21):
            row[hand] = None
            if hand == "Right":
                row[f'x{i}'], row[f'y{i}'], row[f'z{i}'] = None, None, None
            else:
                row[f'X{i}'], row[f'Y{i}'], row[f'Z{i}'] = None, None, None

    #loop throguh 2 hands
    for hand_indx, hand_landmarks in enumerate(result.hand_landmarks):
        #check which hand and the coordinates then append the hand to the row
        which_hand = result.handedness[hand_indx][0].category_name
        if which_hand == "Right":
            row["Right"] = which_hand #this for better visualsiation
            x, y, z = 'x', 'y', 'z'
        else:
            row["Left"] = which_hand
            x, y, z = 'X', 'Y', 'Z'

        #loop throguh each landmark and append it to the dictionary
        for index ,landmark in enumerate(hand_landmarks):
            coords_x, coords_y, coords_z = landmark.x, landmark.y, landmark.z
            row[f'{x}{index}'] = coords_x
            row[f'{y}{index}'] = coords_y
            row[f'{z}{index}'] = coords_z

    #append it to the log
    MOTION_LOG_BUFFER.append(row)
    

def start_timer(seconds=5):
    print(f'starting {seconds} seconds countdown')

    for i in range(seconds, 0, -1):
        print(i)
        time.sleep(1)

    print('Recording Started')
    return True

#write headers
def write_static_header():
    have_header = False

    #create da header and the body data
    header = ["label", 'session_id', "handness"]
    for index in range(21):
        header += [f'x{index}', f'y{index}', f'z{index}']

    #open the file, check if got heading
    file_path = Path("data") / "test_static.csv"

    #check if got data if got means have header
    if os.path.isfile(file_path) and os.path.getsize(file_path) != 0:
        have_header = True

    #write header
    if have_header is False:
        with open("data/test_static.csv", 'a', newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

def write_motion_header():
    have_header = False
    handness = ["Right", "Left"]
    header_row = ["label", "session_id", "clip", "frame"]
    for hand in handness:
        header_row += [hand]
        for index in range(21):
            if hand == "Right":
                header_row += [f'x{index}', f'y{index}', f'z{index}']
            else:
                header_row += [f'X{index}', f'Y{index}', f'Z{index}']
    
    #open file path 
    file_path = Path("data")/ "motion.csv"

    #check if got data
    if os.path.isfile(file_path) and os.path.getsize(file_path) != 0:
        have_header = True
    
    #write header
    if have_header is False:
        with open("data/motion.csv", "a", newline="", encoding='utf') as f:
            writer = csv.writer(f)
            writer.writerow(header_row)

if __name__ == "__main__":
    main()


