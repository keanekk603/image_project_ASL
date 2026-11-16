#test for the neural network
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from features import get_motion_data, get_static_data, save_data

motion_filepath = Path("data")/ "test_motion.csv"
static_filepath = Path("data") / 'test_static.csv'
normalised_motion_filepath = Path("data") / "normalised_motion_test.csv"
normalised_static_filepath = Path("data") / "normalised_static_test.csv"

def main():
    #normalise static data
    static_data = get_static_data(static_filepath)

    #normalise motion data
    motion_data = get_motion_data(motion_filepath)

    #save data
    save_data(motion_data, static_data, normalised_static_filepath, normalised_motion_filepath)

    #run the test
    static_test()
    motion_test()

def static_test():
    static_df = pd.read_csv(normalised_static_filepath)
    columns_to_get = []
    for i in range(21):
        columns_to_get += [f'x{i}', f'y{i}', f'z{i}']
    coordinates = static_df[columns_to_get]
    coordinates = coordinates.to_numpy(dtype=np.float32)

    # get the list of labels and give it its number
    labels = static_df["label"].tolist()
    model_type = "static"
    label_int = get_label(labels, model_type)

    #get the model
    model = tf.keras.models.load_model("models/static_neural_network.keras")

    #test model
    model.evaluate(coordinates, label_int, verbose=2)

#motion test
def motion_test():
    #load the data
    motion_data = pd.read_csv(normalised_motion_filepath)   

    #seperate into thej respecitve recordings
    all_clips = motion_data.groupby(["label", "session_id", "clip"])
    clip_labels = []
    clip_coordinates = []

    headers = []
    for i in range(21):
        headers += [f'x{i}', f'y{i}', f'z{i}', f'X{i}', f'Y{i}', f'Z{i}']

    #loop throguh each df
    for (label, session_id, clip), clip_df in all_clips:
        #append the label onto that
        clip_labels.append(label)

        #get the data in the clip_df file and convert to numpy array
        coordinates = clip_df.sort_values(by="frame")[headers]

        #fill all none/na with 0
        coordinates = coordinates.fillna(0)

        #convert into a 2d array
        coordinates = coordinates.to_numpy(dtype=np.float32)
        clip_coordinates.append(coordinates)
    
    #give a type
    model_type = "motion"

    #convert the labels into int
    clip_label_int = get_label(clip_labels, model_type)
    clip_coordinates = np.array(clip_coordinates)

    #get the model
    model = tf.keras.models.load_model("models/motion_neural_network.keras")

    #test model
    model.evaluate(clip_coordinates, clip_label_int, verbose=2)

#gets whatever labels that was saved from the training
def get_label(labels, model_type):
    #load whatever mode
    if model_type == "static":
        filepath = "data/static_labels.csv"
    elif model_type == "motion":
        filepath = "data/motion_labels.csv"
    else:
        print("wrong data")

    #get the data
    label_data = pd.read_csv(filepath)
    label_values = label_data["label"].tolist()
    label_id = label_data["id"].to_list()
    
    #convert it a dict
    label_to_id = {}
    if len(label_id) == len(label_values):
        length = len(label_id)
    else:
        print("WRONG LENGTH")
    for i in range(length):
        label_to_id[label_values[i]] = label_id[i]

    #convert all of this into the int
    label_int = np.array([label_to_id[label] for label in labels])

    return label_int

if __name__ == "__main__":
    main()