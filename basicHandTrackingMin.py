# this document will cover my first computer vision code. 
# I will be using a palm detection model along with a hand landmarks module to track the hand from the camera
# the palm detection works on the full image and provides a cropped out image with just the hand
# from there the hand landmarks module will mark 20 different points on that cropped hand, for this to be done 
# 30000 images of hands were manually annotated
# this was done with the help of the video https://www.youtube.com/watch?v=NZde8Xt78Iw&t=0s





#we will first import the dependencies
import cv2
import mediapipe as mp
import time

#now we will create the very basic code for running our webcam
cap = cv2.VideoCapture(0)


#this is like formalities that should always be used when doing hand detection model
mpHands = mp.solutions.hands
hands = mpHands.Hands()  # if you want to look at what the parameters that need to be put in are crl+right click on the function
mpDraw = mp.solutions.drawing_utils

#these two are setup for fps calculation
ptime= 0
ctime= 0



#this is the while loop that will continuously put out new frames from the webcam
while True:
    success, img = cap.read()

    
    # very first thing to do is to convert the image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    
    #to check if something is being detected in the results we can print them out this way:
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        # now we will do this separetely for each hand that is being detected:
        for handLms in results.multi_hand_landmarks: # we will extract all the points for each hand and draw the 20 landmarks
            #the coordinates of each landmark on each hand are recorded, and we can access them in the following way
            #each landmark has 2 attributes, id which is 0 through 19 and the coordinate which is given in a ratio
            for id,lm in enumerate(handLms.landmark):
                # print(id,lm)
                #first we will conver the ratio to pixel values
                h,w,c = img.shape  #this gives us the full sizes of the image
                cx , cy = int(lm.x*w) , int(lm.y*h)
                print(id,cx , cy)

                #we can even highlight individual lms now that we know thier coordinates
                if id%2 == 0:
                    cv2.circle(img,(cx,cy),15,(255,255,0),cv2.FILLED)                
            
            
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)  # we will be diplaying img not results so we will draw on img
            
    
    
    #for calculating and displaying the fps
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    #the next part will add the fps on the output window
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,255,0),3)
    
    
    
    
    
    
    
    
    cv2.imshow("image",img) # this will output the image detected by the webcam
    cv2.waitKey(1)#this is important as it tells the program what the delay should be to update the video