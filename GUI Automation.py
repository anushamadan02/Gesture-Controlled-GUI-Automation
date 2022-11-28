import cv2
import imutils
import numpy as np
import sklearn
import webbrowser
from gtts import gTTS 
from sklearn.metrics import pairwise
import pyautogui
import os
import sys

bg = None
def run_avg(image, accumWeight):#larger accumweight good when lot of motion happens
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")#converted to float
        return
    cv2.accumulateWeighted(image, bg, accumWeight)#updates a running average

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]#binary (basically seperated from the background)

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(),
    cv2.RETR_EXTERNAL,#external contours are observed
    cv2.CHAIN_APPROX_SIMPLE)#the inside contours are submerged

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)   #area
        return (thresholded, segmented)

def count(thresholded, segmented):
    # find the convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])#makes tuples or sections
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm
    cX = (extreme_left[0] + extreme_right[0]) // 2
    cY = (extreme_top[1] + extreme_bottom[1]) // 2

    #the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]# the max ED between the center and points of CH
    maximum_distance = distance[distance.argmax()]
    radius = int(0.8 * maximum_distance)#80%of max ED is radius
    circumference = (2 * np.pi * radius)
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")#circular region of palm and fingers
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)#draw the circular roi

   
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)#to neglect the effect of hand lines


    
    (cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#finding contours in the circular roi
    count = 0

    
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        #     25% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1
                

    return count
if __name__ == "__main__":
    # initialize accumulated weight
    accumWeight = 0.5
    camera = cv2.VideoCapture(0)
    template = cv2.imread("face.jpg", cv2.IMREAD_GRAYSCALE)
    
    w, h =template.shape[::-1]
    while True:
        _, frame = camera.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.2)
        #print(loc)
        def Enquiry(loc):
            #print(np.array(loc))
            return(np.array(loc))
        if Enquiry(loc).size:
           
            print('Granted permission!!!')
            mytext = 'Access granted to Anusha'
            language = 'en'
            myobj = gTTS(text=mytext, lang=language, slow=False) 
            myobj.save("output.mp3") 
            os.system("start output.mp3")
            new=2
            url="https://accounts.google.com/signin/v2/identifier?service=youtube&uilel=3&passive=true&continue=https%3A%2F%2Fwww.youtube.com%2Fsignin%3Faction_handle_signin%3Dtrue%26app%3Ddesktop%26hl%3Den%26next%3D%252F&hl=en&ec=65620&flowName=GlifWebSignIn&flowEntry=ServiceLogin";
            webbrowser.open(url,new=new)
            pyautogui.click(332,366,duration=15)#emailid
            pyautogui.typewrite("anushamadangopal.ee17@bmsce.ac.in")
            pyautogui.typewrite(["enter"])
            pyautogui.click(332,366,duration=5)#password
            pyautogui.typewrite("")
            pyautogui.typewrite(["enter"])
            pyautogui.click(436,113,duration=10)#searchbutton
           
            pyautogui.click(198,112,duration=7)#searchbar
            pyautogui.typewrite("bms college of engineering bangalore")
            pyautogui.typewrite(["enter"])
            pyautogui.click(199,328,duration=5)
        

            camera.release()
            break
        else:
            print('Waiting............')
    
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0
    calibrated = False
    while(True):
        (grabbed, frame) = camera.read()#getting current frame
        frame = imutils.resize(frame, width=700)#resize the frame
        frame = cv2.flip(frame, 1)#flipping
        clone = frame.copy()#clone the frame
        (height, width) = frame.shape[:2]#getting height and width o frame
        roi = frame[top:bottom, right:left]#getting region of interest
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        cv2.imshow('ghduwegydq',gray)
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull...")
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))#drawing the contours and displayinfg
                fingers = count(thresholded, segmented)
                while(fingers):
                    if fingers==1:
                        pyautogui.click(78,512,duration=3)#play/pause
                    elif fingers==2:
                        pyautogui.click(174,515,duration=3)#mute/unmute
                    #elif fingers==3:
                        #pyautogui.click(163,591,duration=5)
                    break
                cv2.putText(clone, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                #cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1
        cv2.imshow("Video Feed", clone)
        keypress = cv2.waitKey(1) 


cv2.destroyAllWindows()
camera.release()
