# this program normalises the data into frames of 40 for videos and the photos 

import pandas as pd
from pathlib import Path
import numpy as np
import math
from scipy.interpolate import splprep, splev

FRAME_NORMALISATION = 40
SHAPE = {"x":1920, "y":1080, "z":1920}
motion_filepath = Path("data")/ "motion.csv"
static_filepath = Path("data") / 'landmarks.csv'
normalised_motion_filepath = Path("data") / "normalised_motion.csv"
normalised_static_filepath = Path("data") / "normalised_landmarks.csv"

#convert the data based on scaling 
def main():
    #normalise static data
    static_data = get_static_data(static_filepath)

    #normalise motion data
    motion_data = get_motion_data(motion_filepath)

    #save data
    save_data(motion_data, static_data, normalised_static_filepath, normalised_motion_filepath)

def save_data(motion_data, static_data, static_file, motion_file):
    #save
    motion_data.to_csv(motion_file, index=False)
    static_data.to_csv(static_file, index=False)

#read motion data
def get_motion_data(filepath):
    motion_data = pd.read_csv(filepath)
    for column in motion_data.columns:
        #multiply each column by their pixel
        if column[0].lower() in SHAPE.keys():
            motion_data[column] *= SHAPE[column[0].lower()]

    return normalise_motion(motion_data)

#normalise motion_data
def normalise_motion(motion_data):
    
    #group it by clips, session id and label
    clips = motion_data.groupby(["session_id", "clip", "label"])

    #joint clip for later
    new_clips = []

    #for each clip 
    for clip_num, clip_df in clips:
        
        #list for the clip for frame normalsiation
        new_clip_rows = []

        #refrence row
        reference_row = clip_df.iloc[0]

        #which hand is there
        right_hand = True if pd.notna(reference_row["Right"]) else False
        left_hand = True if pd.notna(reference_row["Left"]) else False
            
        #average distance
        connections = [(0, 5), (0, 17), (0, 9)]
        left_total_distance = 0
        right_total_distance = 0
        for connection in connections:
            i, f = connection[0], connection[1]
            #check if its none, check if its both
            if right_hand and left_hand:
                #if its both hands in the frame we take reference from right side only
                right_total_distance += math.sqrt((reference_row[f'x{i}'] - reference_row[f'x{f}'])**2 + (reference_row[f'y{i}'] - reference_row[f'y{f}'])**2 +(reference_row[f'z{i}'] - reference_row[f'z{f}'])**2)
                left_total_distance = right_total_distance
            else:
                if right_hand:
                    right_total_distance += math.sqrt((reference_row[f'x{i}'] - reference_row[f'x{f}'])**2 + (reference_row[f'y{i}'] - reference_row[f'y{f}'])**2 +(reference_row[f'z{i}'] - reference_row[f'z{f}'])**2)

                if left_hand:
                    left_total_distance += math.sqrt((reference_row[f'X{i}'] - reference_row[f'X{f}'])**2 + (reference_row[f'Y{i}'] - reference_row[f'Y{f}'])**2 +(reference_row[f'Z{i}'] - reference_row[f'Z{f}'])**2)

        #if its none the average would be 0
        left_average = left_total_distance/3
        right_average = right_total_distance/3

        #reference point right and left, if both hands take reference to right, this in itself already accounts for the distance between hands
        if right_hand and left_hand:
            ref_x, ref_y, ref_z = reference_row["x0"], reference_row["y0"], reference_row['z0']
            ref_X, ref_Y, ref_Z = ref_x, ref_y, ref_z
        else:
            if right_hand:
                ref_x, ref_y, ref_z = reference_row["x0"], reference_row["y0"], reference_row['z0']
            if left_hand:
                ref_X, ref_Y, ref_Z = reference_row["X0"], reference_row["Y0"], reference_row['Z0']

        #loop throguh each row in the clip
        for row in clip_df.itertuples(index=False):
            new_row = row._asdict()
            for i in range(21):
                #normalise it by the frame one coordinates and scale it 
                if right_hand:
                    new_row[f'x{i}'] = (getattr(row, f'x{i}') - ref_x)/right_average
                    new_row[f'y{i}'] = (getattr(row, f'y{i}') - ref_y)/right_average
                    new_row[f'z{i}'] = (getattr(row, f'z{i}') - ref_z)/right_average

                if left_hand:
                    new_row[f'X{i}'] = (getattr(row, f'X{i}') - ref_X)/left_average
                    new_row[f'Y{i}'] = (getattr(row, f'Y{i}') - ref_Y)/left_average
                    new_row[f'Z{i}'] = (getattr(row, f'Z{i}') - ref_Z)/left_average
            
            #add it to the 2d list
            row_to_add = list(new_row.values())
            new_clip_rows.append(row_to_add)

        #no. of frames trim 5 based on result test
        number_of_frames = len(new_clip_rows) - 5

        #check if number of frames > 13 kinda means its like a
        if number_of_frames >= 13:
            #create a new dataframe
            new_clip_df = pd.DataFrame(new_clip_rows[:number_of_frames], columns=clip_df.columns)

            #start on intraporlation/drop
            if number_of_frames != FRAME_NORMALISATION:
                new_clip = data_frame_change(new_clip_df, right_hand, left_hand)
            else:
                new_clip = new_clip_df
            

            #combine the rows to the new data frame
            new_clips += new_clip.values.tolist()
        
    
    #return the list of rows
    return pd.DataFrame(data=new_clips, columns=motion_data.columns)


#dropped(too much) and intrapolation(too little datapoints)
def data_frame_change(motion_df, right_hand, left_hand):
    #get first row
    first_row = motion_df.iloc[0].to_dict()

    #make a header bascially
    new_dict = dict.fromkeys(motion_df.columns)
    new_dict["label"], new_dict["session_id"], new_dict["clip"], new_dict["Right"], new_dict["Left"] = first_row["label"], first_row["session_id"], first_row["clip"], first_row["Right"], first_row["Left"]

    #loop throguh each coordinate point starting with point 0
    for i in range(21):
        x_right, y_right, z_right, x_left, y_left, z_left = [], [], [], [], [], []

        #loop throguh each row, and check if theres a right or left hand then append it to the coordinate list
        if right_hand:
            x_right, y_right, z_right = motion_df[f'x{i}'], motion_df[f'y{i}'], motion_df[f'z{i}']
        if left_hand:
            x_left, y_left, z_left = motion_df[f'X{i}'], motion_df[f'Y{i}'], motion_df[f'Z{i}']
        
        """
            1. splpre: Spline Parametric Representation
            - analyses raw data and build mathematical model of the shape, calcualte path distance throguh 3D space (x, y, z) and figures how the curve bends between points
            - inputs([list of raw coordinates], k = spline degree (2 for 3 datapoints), s=smootihing factor, per=(whether its a loop))
            - k = mathamtical complexity of the curves used to connect data points
            - s = the value that controls how stric the line must follow your raw tracking data, the higher the less strict
            - outputs return tck(mathematical blueprint tuple containing knots coefficients and degree) and u (the original positions from 0.0 to 1.0)
            - u an array of values from 0 - 1.0 that acts as a parametric timeline
            - tck: compiled mathematicalblueprint contianing everything needed to reconstruct smooth curve

            2. np.linspace: linear space
            - used to generate a sequaence of evenly spaced out numbers over a specificed mathematical interval
            - 0 - 1 is the parametric timing
            - 40 is the maximum number of datapoints

            3. splev: Spline Evaluation
            - calculates the actual smooth coordinate numbers
            input:
            - u_new your 40 step parametric timeline generated by linspace
            - tck the mathematical blueprint for connecting the knots
            - der (derative) if der = 0(position), der=1(velocity), der=2(acceleration)
            """
        #interpolate right side
        if len(x_right) != 0 or len(y_right) != 0 or len(z_right) != 0:
            tck, u = splprep([np.array(x_right), np.array(y_right), np.array(z_right)], k=2, s=0.1, per=False)
            u_new = np.linspace(0, 1, FRAME_NORMALISATION)
            x_right, y_right, z_right = splev(u_new, tck, der=0)

        #interpolate left side
        if len(x_left) != 0 or len(y_left) != 0 or len(z_left) != 0:
            tck, u = splprep([np.array(x_left), np.array(y_left), np.array(z_left)], k=2, s=0.1, per=False)
            u_new = np.linspace(0, 1, FRAME_NORMALISATION)
            x_left, y_left, z_left = splev(u_new, tck, der=0)

        #write the data in 
        if len(x_right) != 0 and len(y_right)!= 0 and len(z_right)!= 0:
            new_dict[f"x{i}"], new_dict[f"y{i}"], new_dict[f"z{i}"] = x_right, y_right, z_right

        if len(x_left) != 0 and len(y_left)!= 0 and len(z_left) != 0:
            new_dict[f"X{i}"], new_dict[f"Y{i}"], new_dict[f"Z{i}"] = x_left, y_left, z_left

    return convert_to_df(new_dict, right_hand, left_hand)

#convert the new dict to dataframe
def convert_to_df(new_dict, right_hand, left_hand):
    #check if got hands
    no_of_frames = 0
    if right_hand:
        no_of_frames = len(new_dict["x0"])
    else:
        if left_hand:
            no_of_frames = len(new_dict["X0"])

    #make the number of items in each column equal
    if no_of_frames != 0:
        new_dict['label'] = [new_dict['label']] * no_of_frames
        new_dict['session_id'] = [new_dict['session_id']] * no_of_frames
        new_dict['clip'] = [new_dict['clip']] * no_of_frames
        new_dict['frame'] = [i for i in range(1, no_of_frames + 1)]
        if not right_hand:
            new_dict["Right"] = [None] * no_of_frames
            for i in range(21):
                new_dict[f'x{i}'], new_dict[f'y{i}'], new_dict[f'z{i}'] = [None] * no_of_frames, [None] * no_of_frames, [None] * no_of_frames
        else:
            new_dict["Right"] = ["Right"] * no_of_frames
        if not left_hand:
            new_dict["Left"] = [None] * no_of_frames
            for i in range(21):
                new_dict[f'X{i}'], new_dict[f'Y{i}'], new_dict[f'Z{i}'] = [None] * no_of_frames, [None] * no_of_frames, [None] * no_of_frames
        else:
            new_dict["Left"] = ["Left"] * no_of_frames
    
    #build a new dataframe and return it
    return pd.DataFrame(new_dict)

#read static data
def get_static_data(filepath):
    #load statick file path
    static_data = pd.read_csv(filepath)
    for column in static_data.columns:
        if column[0].lower() in SHAPE.keys():
            static_data[column] *= SHAPE[column[0].lower()]

    #create new data table and input the tables
    new_rows = []
    for row in static_data.itertuples(index=False):
        new_row = normalise_static(row)
        new_rows.append(new_row)
    
    new_data = pd.DataFrame(new_rows, columns=static_data.columns)
        
    return new_data

def normalise_static(row):
    #scaling distance (0,5) (0, 17), (0,9)
    connections = [(0,5), (0,17), (0,9)]
    total_distance = 0

    #get average of 3 distance and divide by 3
    for connection in connections:
        i, f = connection[0], connection[1]
        distance = math.sqrt((getattr(row, f"x{i}") - getattr(row, f"x{f}"))**2 + (getattr(row, f"y{i}") - getattr(row, f"y{f}"))**2 +(getattr(row, f"z{i}") - getattr(row, f"z{f}"))**2)
        total_distance += distance

    average_distance = total_distance/3

    #converts the values into a list
    row = list(row)
    unchanged, coordinates, new_row = row[:3], row[3:], []
    reference_coords = (row[3], row[4], row[5]) #normalise to the first ccoords x0, y0, z0
    new_row += unchanged

    #loop throguh cooridnates and minus off, reference
    for i in range(21):
        new_row += [(coordinates[i * 3] - reference_coords[0])/average_distance, (coordinates[i * 3 + 1] - reference_coords[1])/average_distance, (coordinates[i * 3 + 2] - reference_coords[2])/average_distance]

    return new_row
if __name__ == "__main__":
    main()

"""
itertuples much faster
can call thigns lie row.x, row.y, row.z or use an getattr(row, x)
different stuff like pd.groupby dont always create new dataframe for fun lmao
aso b asicallt we normalsie to the position of the first frame the first poiint 0 and the distance is the average distance between the fingers

if both hands are present we take reference tot he right hand
so this means that the scale is based on the average distance in the right hand
and we are subtracting from the right hand points x y z
"""