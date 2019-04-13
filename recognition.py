'''
PackHacks Rock Paper Scissors

A computer-vision based version of rock-paper-scissors

'''

# Credit for gesture recognition tutorial: https://gogul09.github.io/software/hand-gesture-recognition-p1

import cv2
import numpy as np
from keras.models import load_model

bg = None
response = ''

# Find the average of the background
def run_avg(image, weight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, weight)

# Function to distinguish the hand from the background
def segment(image, threshold=10):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    thresholded = cv2.GaussianBlur(thresholded,(5,5),0)
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return None
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

if __name__ == "__main__":
    # Loading our CNN's model
    model = load_model("model.h5")

    accumWeight = 0.5
    im_count = 0

    camera = cv2.VideoCapture(0)

    x, y, r = 300, 300, 200
    # region of interest (ROI) coordinates
    top, right, bottom, left = x-r, y-r, x+r, y+r

    num_frames = 0

    while(True):
        # Start grabbing frames each iteration
        (grabbed, frame) = camera.read()
        frame = cv2.flip(frame, 1)
        clone = frame.copy()

        # Establishing height and width and setting the region of interest
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # establish a constant background 
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 29:
                # Console output
                print("Ready to go!")
                import time
                print("Rock!")
                time.sleep(1)
                print("Paper!")
                time.sleep(1)
                print("Scissor!")
                time.sleep(1)
                print("Shoot!")
                time.sleep(1)
        else:
            # segment the hand region
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                ep = 0.01*cv2.arcLength(segmented,True)
                segmented = cv2.approxPolyDP(segmented,ep,True)

                convex_hull = cv2.convexHull(segmented)

                # Visualizing our ROI and showing contours
                cv2.rectangle(clone, (left, top), (right, bottom), (0,0,0), thickness=cv2.FILLED)
                cv2.drawContours(clone, [convex_hull + (right, top)], -1, (213, 187, 25), thickness=cv2.FILLED)
                cv2.drawContours(clone, [segmented + (right, top)], -1, (14, 41, 60), thickness=cv2.FILLED)

                # Running frame through our neural network
                preds = model.predict(cv2.resize(clone[top:bottom, right:left], (64, 64)).reshape((-1, 64, 64, 3)))[0]
                index = np.argmax(preds)
                
                response = ["rock", "paper", "scissor"][index]
                break
               
        cv2.rectangle(clone, (left, top), (right, bottom), (25, 187, 213), 2)

        # increment the number of frames
        num_frames += 1
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break

    # Computer move
    import random
    comp_response = ["rock", "paper", "scissor"][random.randint(0,2)]

    print("Computer chose: " + comp_response)
    print("Player chose: " + response)

    # Game Logic
    if(response == comp_response):
        print("Tie!")
    elif((response == 0 and comp_response == 2) or (response == 1 and comp_response == 0) or (response == 2 and comp_response == 1)):
        print("Player Wins!")
    else:
        print("Computer Wins!")

    # free up memory
    camera.release()
    cv2.destroyAllWindows()
