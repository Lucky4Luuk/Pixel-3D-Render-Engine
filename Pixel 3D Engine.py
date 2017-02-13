import pygame
from pygame.locals import *
import cv2
import sys
import numpy
from OpenGL.GL import *
from OpenGL.GLU import *

WIDTH,HEIGHT = 800,600

cascPath = "haarcascade_frontalface_default.xml.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

pygame.init()
pygame.display.set_mode([WIDTH,HEIGHT],DOUBLEBUF|OPENGL)
gluPerspective(45, (WIDTH/HEIGHT), 0.1, 50.0)
glTranslatef(0.0,0.0,-5)
glRotatef(0,0,0,0)

while True :
    #Clear Window
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    #Handle Webcam stuff
    if not video_capture.isOpened() :
        print("Unable to load camera.")
        raise OSError("Unable to load camera.")
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30)
    )
    
    # Draw a rectangle around the faces
    try :
        for (x,y,w,h) in faces :
            sx = (x-WIDTH/2)/50
            sy = (y-HEIGHT/2)/50
            glBegin(GL_QUADS)
            glColor3f(1,1,1)
            glVertex3f(sx-1,sy-1,-1)
            glVertex3f(sx-1,sy+1,-1)
            glVertex3f(sx+1,sy+1,-1)
            glVertex3f(sx+1,sy-1,-1)
            glEnd()
    except Exception as e :
        print(e)

    #Handle Events
    for event in pygame.event.get() :
        if event.type == pygame.QUIT :
            pygame.quit()
            exit()
    
    #Update Window
    pygame.display.flip()

#Close window
pygame.quit()

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

#Exit Program
exit()
