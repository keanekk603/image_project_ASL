# need 2 models one for static and one for motion
from pathlib import Path
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

EPOCHS = 50
SEED = 123
#initial number of categories 0
MOTION_CATEGORIES = 0
STATIC_CATEGORIES = 0
VALIDATION_SPLIT = 0.30
NUM_FEATURES = 63

MOTION_DATA_PATH = Path('data/normalised_motion.csv')
STATIC_DATA_PATH = Path('data/normalised_landmarks.csv')

def main():
    static_data = pd.read_csv(STATIC_DATA_PATH)
    motion_data = pd.read_csv(MOTION_DATA_PATH)
    
    create_static_neural_network(static_data)
    create_motion_neural_network(motion_data)

#motion model
def create_motion_neural_network(motion_data):
    motion_coordinates, motion_labels = load_motion_data(motion_data)

    #leave teh clip out 
    input_shape = motion_coordinates.shape[1:]

    #split motion test and training data
    motion_x_train, motion_x_test, motion_y_train, motion_y_test = train_test_split(
        motion_coordinates, motion_labels, test_size=VALIDATION_SPLIT, stratify=motion_labels, random_state=SEED
    )

    #get motion model
    motion_model = get_motion_model(input_shape)

    #fit model
    motion_model.fit(
        motion_x_train, motion_y_train,
        validation_data=(motion_x_test, motion_y_test),
        epochs=EPOCHS,
        batch_size=32
    )

    #test it 
    motion_model.evaluate(motion_x_test, motion_y_test, verbose=2)

    #save model
    motion_model.save(Path("models/motion_neural_network.keras"))

#load motion data
def load_motion_data(motion_data):
    #seperate into the respecitve recordings
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

    #converts it into a 3d array
    return np.array(clip_coordinates), clip_label_int

#motion_model
def get_motion_model(input_shape):
    model = tf.keras.models.Sequential([
        #inputs we leave the number fo clips out, this just tell the model the shape of a unit input
        tf.keras.Input(shape=input_shape),

        #how many neurons, reads 40 frames in order, Long short term memory is a layer built for sequences:
        #where the order of the data matters
        #its a different kind of layer a recurrent layer go throguh the input step by step and carry the memory forward so it will understand the order
        tf.keras.layers.LSTM(32, activation="relu"),

        #dropout (0.2)
        tf.keras.layers.Dropout(0.2),

        #hidden layers
        tf.keras.layers.Dense(16, activation="relu"),

        #output layer
        tf.keras.layers.Dense(MOTION_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy", #integer labels
        metrics=["accuracy"], 
    )
    
    return model

#static model
def create_static_neural_network(static_data):
    static_coordinates, static_labels = load_static_data(static_data)

    #split static data in test and train
    static_x_train, static_x_test, static_y_train, static_y_test = train_test_split(
        static_coordinates, static_labels, test_size=VALIDATION_SPLIT, stratify=static_labels, random_state=SEED
    )

    #get the static model
    static_model = get_static_model()

    #fit the model
    static_model.fit(
        static_x_train, static_y_train,
        validation_data=(static_x_test, static_y_test),
        epochs = EPOCHS,
        batch_size=32,
    )
    
    #test it
    static_model.evaluate(static_x_test, static_y_test, verbose=2)

    #save_static model
    static_model.save(Path("models/static_neural_network.keras"))

def get_static_model():
    model = tf.keras.models.Sequential([
        #inputs, leave the number of rows out thats why only NUM_FEATURES, we want to give them the shape of a unit of input which is 63 values  and one row only
        tf.keras.Input(shape=(NUM_FEATURES, )),
        
        #how many layers
        tf.keras.layers.Dense(128, activation="relu"),

        #dropout (0.2)
        tf.keras.layers.Dropout(0.2),

        #hidden layers
        tf.keras.layers.Dense(64, activation="relu"),

        #output layer
        tf.keras.layers.Dense(STATIC_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy", #integer labels
        metrics=["accuracy"], 
    )

    return model


def load_static_data(static_df):
    #get the list of rows
    columns_to_get = []
    for i in range(21):
        columns_to_get += [f'x{i}', f'y{i}', f'z{i}']
    coordinates = static_df[columns_to_get]
    coordinates = coordinates.to_numpy(dtype=np.float32)

    # get the list of labels and give it its number
    labels = static_df["label"].tolist()
    model_type = "static"
    label_int = get_label(labels, model_type)
    return coordinates, label_int

#saves the data in another file
def get_label(labels, model_type):
    #initalsie the num of categories
    global MOTION_CATEGORIES, STATIC_CATEGORIES

    label_to_id = {}
    set_labels = sorted(set(labels))

    #make sure the set is sorted so its sorted alphabetically and th4e same order always because set alwauys change
    for index, label in enumerate(set_labels):
        label_to_id[label] = index

    #convert all of this into the int
    label_int = np.array([label_to_id[label] for label in labels])

    #label df
    label_keys = list(label_to_id.keys())
    label_values = list(label_to_id.values())
    dataframe = {"label": label_keys, "id": label_values}
    label_df = pd.DataFrame(dataframe)

    #check the type 
    if model_type == "static":
        STATIC_CATEGORIES = len(label_values)
        label_df.to_csv("data/static_labels.csv", index=False)
    elif model_type == "motion":
        MOTION_CATEGORIES = len(label_values)
        label_df.to_csv("data/motion_labels.csv", index=False)

    return label_int

if __name__ == "__main__":
    main()
    

