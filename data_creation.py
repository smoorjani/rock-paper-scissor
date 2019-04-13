'''
PackHacks Rock Paper Scissors

A computer-vision based version of rock-paper-scissors
'''

# Gesture recognition tutorial: https://gogul09.github.io/software/hand-gesture-recognition-p1

import cv2
import numpy as np
from keras.models import load_model

bg = None

def run_avg(image, weight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, weight)

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
    model = load_model("model.h5")
    accumWeight = 0.5

    im_count = 0

    camera = cv2.VideoCapture(0)

    x, y, r = 300, 300, 200
    # region of interest (ROI) coordinates
    top, right, bottom, left = x-r, y-r, x+r, y+r

    num_frames = 0

    while(True):
        (grabbed, frame) = camera.read()
        frame = cv2.flip(frame, 1)
        clone = frame.copy()

        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 29:
                print("Ready to go!")
        else:
            # segment the hand region
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                ep = 0.01*cv2.arcLength(segmented,True)
                segmented = cv2.approxPolyDP(segmented,ep,True)

                convex_hull = cv2.convexHull(segmented)

                cv2.rectangle(clone, (left, top), (right, bottom), (0,0,0), thickness=cv2.FILLED)
                cv2.drawContours(clone, [convex_hull + (right, top)], -1, (0, 255, 0), thickness=cv2.FILLED)
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255), thickness=cv2.FILLED)

                preds = model.predict(cv2.resize(clone[top:bottom, right:left], (64, 64)).reshape((-1, 64, 64, 3)))[0]
                index = np.argmax(preds)
                
                text = ["rock", "paper", "scissor"][index] + " " + str(round(preds[index], 2))
                print(text)
               
        cv2.rectangle(clone, (left, top), (right, bottom), (255,0,0), 2)

        # increment the number of frames
        num_frames += 1
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break
        path = None
        # Creating data files based on user input
        if keypress == ord("r"):
            path = "r" + str(im_count) + ".png"
        elif keypress == ord("p"): 
            path = "p" + str(im_count) + ".png"
        elif keypress == ord("s"):
            path = "s" + str(im_count) + ".png"

        if path is not None:
            cv2.imwrite("data/" + path, clone[top:bottom, right:left])
            print ("saved", path)
            im_count += 1

    # free up memory
    camera.release()
    cv2.destroyAllWindows()
