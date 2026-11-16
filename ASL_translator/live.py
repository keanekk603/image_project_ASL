import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import pandas as pd
from features import normalise_motion, normalise_static
import numpy as np
import tensorflow as tf
from copy import deepcopy

#create 
class Translator():
    def __init__(self):
        self.static_network = tf.keras.models.load_model("models/static_neural_network.keras")
        self.motion_network = tf.keras.models.load_model("models/motion_neural_network.keras")
        self.motion_label = self.load_motion_label()
        self.static_label = self.load_static_label()
    
    def load_motion_label(self):
        #convert to a dictionary
        motion_df = pd.read_csv("data/motion_labels.csv")
        labels, id = motion_df["label"].tolist(), motion_df["id"].tolist()

        #number of id, new dictionary
        num_ids = len(labels)
        id_to_labels = {}

        #from the parallel list create a dictionary
        for i in range(num_ids):
            id_to_labels[id[i]] = labels[i]
        
        return id_to_labels
    
    def load_static_label(self):
        #convert to a dictionary
        motion_df = pd.read_csv("data/static_labels.csv")
        labels, id = motion_df["label"].tolist(), motion_df["id"].tolist()

        #number of id, new dictionary
        num_ids = len(labels)
        id_to_labels = {}

        #from the parallel list create a dictionary
        for i in range(num_ids):
            id_to_labels[id[i]] = labels[i]

        return id_to_labels

class Global():
    def __init__(self):
        self.motion_mode = False
        self.FRAME_COUNTER = 0
        self.START_RECORDING = False
        self.MOTION_LOG_BUFFER = []
        self.total_sentence = []
        self.static_word = []

    def clear_buffer(self):
        self.MOTION_LOG_BUFFER = []
        self.FRAME_COUNTER = 0
    
    def start_motion(self):
        if self.motion_mode is False and self.START_RECORDING is False:
            self.motion_mode = True
            return True
        return False
    
    def end_motion(self):
        if self.motion_mode is True and self.START_RECORDING is False:
            self.motion_mode = False
            return True
        return False

    def start_recording(self):
        if self.motion_mode is True and self.START_RECORDING is False:
            self.START_RECORDING = True
            return True 
        return False
    
    def end_recording(self):
        if self.motion_mode is True and self.START_RECORDING is True:
            self.START_RECORDING = False
            return True
        return False
    
    def record(self, detection):
        if self.START_RECORDING is True and detection:
            return True
        return False
    def clear_static_word(self):
        self.static_word = []

global_state = Global()
translator = Translator()

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

SHAPE = {"x":1920, "y":1080, "z":1920}

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

            #pass the key in
            if key == ord('0'):
                #if not motion mode then can run static mode
                if global_state.motion_mode is False and detection_result.hand_landmarks:
                    predict_static(detection_result)
                pass
            if key == ord('2'):
                global_state.total_sentence.append(global_state.static_word)
                global_state.clear_static_word()

            #record motion
            if key == ord('1'):
                #if mode is false set to true
                if global_state.start_motion():
                    print("motion mode")

                #if mode is true set to false
                else:
                    if global_state.end_motion():
                        print("exiting motion mode")

            #if start reocrdinf through get record motion and if got hand and if the recording is true
            if global_state.record(detection_result.hand_landmarks):
                motion_record(detection_result)

            #start recording
            if key == ord(' '):
                if global_state.start_recording():
                    if start_timer():
                        pass
                
                #reset all the stuff
                else:
                    if global_state.end_recording():
                        predict_motion()
                        global_state.clear_buffer()
                        print("recording stop")
            
            if key == 27: # Esc quits
                break

    print_sentence()
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

def predict_static(result):
    datas = []
    # can have both hand will append first hand that it picks than second
    for hand_indx, hand_landmark in enumerate(result.hand_landmarks):
        #body data
        handness = result.handedness[hand_indx][0]
        hand_data = ["predict", "predict", handness]
        for landmark in hand_landmark:
            hand_data += [landmark.x, landmark.y, landmark.z]
        
        #create da header and the body data
        header = ["label", 'session_id', "handness"]
        for index in range(21):
            header += [f'x{index}', f'y{index}', f'z{index}']

        static_df = pd.DataFrame(data=[hand_data], columns=header)

        #get the pixel first multiply by the shape
        for column in static_df.columns:
            if column[0].lower() in SHAPE.keys():
                static_df[column] *= SHAPE[column[0].lower()]

        # get the data and normalise it
        new_row = []
        for row in static_df.itertuples(index=False):
            new_row = normalise_static(row)

        #just check if the new_row is 63 long
        datas.append(new_row[3:])
    
    #loop throguh each hand data, shld have a set of 63, 2
    model = translator.static_network
    label = translator.static_label
    predictions = model.predict(np.array(datas))
    for prediction in predictions:
        output = label[np.argmax(prediction)]
        
        #append it to the sentence
        global_state.static_word.append(output)
    
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
def predict_motion():
    #write the data in motion log buffer into the motion file
    motion_df = pd.DataFrame(global_state.MOTION_LOG_BUFFER)
    for column in motion_df.columns:
        #multiply each column by their pixel
        if column[0].lower() in SHAPE.keys():
            motion_df[column] *= SHAPE[column[0].lower()]

    #normalise it
    new_df = normalise_motion(motion_df)
    new_df = new_df.fillna(0)

    #row to keep, we trained the model intertwined so im jsut gg do this like that, since the test and the train uses it like that already
    headers = []
    for i in range(21):
        headers += [f'x{i}', f'y{i}', f'z{i}', f'X{i}', f'Y{i}', f'Z{i}']

    #fill na and drop the random columns
    new_df = new_df[headers]

    #convert to numpy, currently its in the format (126, 40,)
    coordinates = new_df.to_numpy(dtype=np.float32)

    #predict model it returns a proabbiltiy list like [[0.01, 0.98, 0.01], []] etc.
    motion_model = translator.motion_network
    #since the model expects a stack fo clips of shape(number of clips, 40, 126) we need to change it to (1, 40, 126)
    #to do this we write this expand dimension and whcih axis, where 0 is the outtest level, stack 1 clip at the outtest level, for axis 1 is stack on frame, axis 2 is the inner most level the numbers in each frame
    coordinates= np.expand_dims(coordinates, axis=0)
    probability_list = motion_model.predict(coordinates).flatten()

    #get the index of the max prob and the output
    max_prob_index = np.argmax(probability_list)
    output = translator.motion_label[max_prob_index]

    #add to the global state
    global_state.total_sentence.append(output)

def motion_record(result):
    #increment the counter
    global_state.FRAME_COUNTER += 1

    #iniitliase a row
    row = {"label": "predict", "session_id": "predict", "clip": "predict", "frame": global_state.FRAME_COUNTER}

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
    global_state.MOTION_LOG_BUFFER.append(row)

def start_timer(seconds=5):
    print(f'starting {seconds} seconds countdown')

    for i in range(seconds, 0, -1):
        print(i)
        time.sleep(1)

    print('Recording Started')
    return True

def print_sentence():
    sentence = global_state.total_sentence #its a list
    deep_copy = deepcopy(sentence)
    for i in range(len(sentence)):
        if isinstance(sentence[i], list):
            deep_copy[i]= "".join(deep_copy[i])

    
    new_sentence = " ".join(deep_copy)
    print(new_sentence)
    success, frame = cap.read()
    cv2.putText(frame, new_sentence, (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ASL Translator", frame)

if __name__ == "__main__":
    main()

