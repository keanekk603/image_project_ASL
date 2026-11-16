#just a visualisation script so u can see if it works
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import pandas as pd
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

def main():
    valid_rows = []

    #read file as panda
    file = pd.read_csv("landmarks.csv")
    valid_rows = file[file['label'] == "D"] #filter where the column is A

    if len(valid_rows) == 0:
        print('no data')
    else:
        x_coordinates = []
        y_coordinates = []
        
        #read the row throguh iterrow pd way
        for index, row in valid_rows.iterrows():
            #grab the column
            x_coordinate = []
            y_coordinate = []
            for i in range(21):
                #add it to the coordinate row[column label]
                x_coordinate.append(row[f'x{i}'])
                y_coordinate.append(row[f'y{i}'])
            
            x_coordinates.append(x_coordinate)
            y_coordinates.append(y_coordinate)

    x_coords_toplot, y_coords_toplot = x_coordinates[2], y_coordinates[2]
    plot_data(x_coords_toplot, y_coords_toplot)


def plot_data(x_coord, y_coord):
    #create the labels
    x_label, y_label = "X", "Y"
    x_coord, y_coord = [float(x) for x in x_coord], [float(y) for y in y_coord]

    #plot the points, scatter plot
    plt.figure(figsize=(8, 5)) # figure size standard
    size = len(x_coord) #number of points
    for i in range(size):
        plt.scatter(x_coord[i], y_coord[i], marker='o', color='b') #plt.plot(x, y, shape of point)

    #draw da data lines, for each connection draw a line
    for start, end in HAND_CONNECTIONS:
        plt.plot(
            [x_coord[start], x_coord[end]], # x start to x end
            [y_coord[start], y_coord[end]], # y start to y end 
            color='b',
            linewidth=2,
            linestyle='-',
            zorder=1 #used for scatter plots
        )
    
    #apply the labels
    plt.xlabel = x_label
    plt.ylabel = y_label

    #draw the grid
    plt.grid(True, linestyle='--', alpha=0.6) #plt.grid(True(means have grid), linestyle='--'(means dashed lines), alpha(refers to opactity))
    plt.legend()
    plt.show() #shows the data

main()
            