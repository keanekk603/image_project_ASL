1. record_data.py (can change the file path)
- try recording some data, press one for motion mode, then key in the data
- then press space to start recording and space to stop recording
- it automatically normalises to 40 frames, cuts out the last 8 frames (accounts for the pressing of the button)
- remmeber to change the file name in the record_data.py, (to record data for trainig/ test)

2. features.py (cannot change the filepath)
- 1st step: all is multipled byt he widht and height for the camera, adjust the width and height respectively based on ur own camera 
- how to:
import cv2

image = cv2.imread('your_image.jpg')
height, width = image.shape[:2]

print("Width:", width)
print("Height:", height)
- then adjust it first

- 2nd step: static normalisation
- for static, it normalises to the first coordinate point x0, y0, z0 everythibng will minus that
- then it scales to 3 hand connections (i forgot), i think its the wrist bones, basically i divide all the points by the average of th4e distance of the 3 wrist bones since they dont change much
- this eliminates the change in depth abcially how far u move ur hand away etc.
- then it creates a new cv file

- 3rd step: video normalisation
- normalised tot he first frame x0, y0, z0 and the first frame average distance too
- this make it such that it accounts for the movement forward and backwards of the hand
- aso it cuts off 8 frames at the end accounts for the time taken to press the space bar
- then it scales to 40 frames wheter it drop points or adds points using some prep method i found online lmao, aso the smooth ratio is 0.01 because we want it to conenc the points well

- 4th step: lastly it saves to 2 seperate file for train.py 

- additional step: u might see a motion_labels and static_labels, its cause it saves the labels that are in the dataset, and its in the correct order so the label to id change is always maintained so when we nromalsie the test data we the label to id conversion is maintained as the ones in used in the neural netowkr training


3. train.py (cannot change the filepath)
- trains teh neural netowkr, the filepath should not be changed in this script
- creates 2 neural network one for static one for motion
- the motion one uses LSTM which basically its a recurrent neural netwokr
- Dense: feedforward neural network (used in static), information stricktly flow in one direction no memory, used for static data where ortder doesnt matter

- LSTM: Recurrent Neural network, has a looping connection that feeds the output of a step back into the neural netowkr alongisde the same input basically, it feeds ouput and input to the neural network again
- theres a memorry of an internal hidden state that acts as a short term memory, this allows the model to remmeber what happen a while ago, used for sequential or time series data where the order matters

4. run the test.py (cannot change the filepath)
- basically gets the test data (the test.csv datas)
- goes throguh it
- normalises all of it the same way features.py normalises the data
- run throguh the neural network and gets the data

Test1:
static_neural network: 0.981 accuracy
motion_neural network: 1.00 accuracy (mainly cause i only got 2 ouput lmao)
go record some stuff and try it out

5. visualisation.py
- wihtin the data folder theres a script that u can see how the computer sees the image
- run it on some random coordinates in the static file u shld see a layout of an inverted hand

6. live.py
- first run the mediapipe script,
- next add a key to start the output, motion or static
- next then once i get the key same thing 3 2 1 
- then once it appends to the output once the file cuts it writes out something

- to detect motion press 1 then space it say 5 4 3 2 1, then u do ur motion then u press space to stop then if u want contiinue another motion then u press space again, to get out of motion mode u press 1

- to detect static u must get out of nmotion mode first then press 0 everytime u make a sign, then it appends to a list a static list of words, then once u made ur own custom word u press 2 to save it into the final sentence, then u can press 0 again to amke anotehr list ah



