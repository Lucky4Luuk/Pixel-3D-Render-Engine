import pygame
from pygame.locals import *
import cv2
import sys
import numpy
from OpenGL.GL import *
from OpenGL.GLU import *
import ctypes

myDLLlib = ctypes.cdll.LoadLibrary("C++ DLLS/DLL Library Project.dll")
print(myDLLlib.sum(5,3))

WIDTH,HEIGHT = 800,600
oldFaceLoc = [0,0]
newFaceLoc = [0,0]
faceLerp = 0
faceFrameMax = 6
faceFrame = faceFrameMax #If faceFrame == faceFrameMax it will update oldFaceLoc

def lerpVec2(a,b,t) :
    newx = (1-t)*a[0] + t*b[0]
    newy = (1-t)*a[1] + t*b[1]
    return [newx,newy]

def lerpVec3(a,b,t) :
    newx = (1-t)*a[0] + t*b[0]
    newy = (1-t)*a[1] + t*b[1]
    newz = (1-t)*a[2] + t*b[2]
    return [newx,newy,newz]

cascPath = "haarcascade_frontalface_default.xml.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
firstret, firstframe = video_capture.read()
camWidth = int(firstframe.shape[0])
camHeight = int(firstframe.shape[1])
aspectW = WIDTH/camWidth
aspectH = HEIGHT/camHeight
drawFace = 1
camx = 0
camy = 0
camz = 1
rotz = 0

pygame.init()
pygame.display.set_mode([WIDTH,HEIGHT],DOUBLEBUF|OPENGL)
gluPerspective(70, (WIDTH/HEIGHT), 1, 500.0)
glTranslatef(0.0,0.0,-5)
glRotatef(0,0,0,0)

class Model() :
    def __init__(self,filename) :
        self.vertices = []
        self.faces = []
        self.load("models/"+str(filename))

    def load(self,filename) :
        self.file = open(filename)
        for line in self.file :
            if line.startswith("v ") :
                vdata = line.split("v ")[1].split(" ")
                x = float(vdata[0])
                y = float(vdata[1])
                z = float(vdata[2])
                self.vertices.append((x,y,z))
            elif line.startswith("f ") :
                fdata = line.split("f ")[1].split(" ")
                try :
                    v1 = int(fdata[0].split("//")[0])-1
                    v2 = int(fdata[1].split("//")[0])-1
                    v3 = int(fdata[2].split("//")[0])-1
                    self.faces.append([v1,v2,v3])
                except Exception :
                    v1 = int(fdata[0].split("/")[0])-1
                    v2 = int(fdata[1].split("/")[0])-1
                    v3 = int(fdata[2].split("/")[0])-1
                    self.faces.append([v1,v2,v3])
        self.file.close()

    def draw(self) :
        glBegin(GL_TRIANGLES)
        for f in self.faces :
            glColor3f(0.5,0.8,0.2)
            glVertex3fv(self.vertices[f[0]])
            glVertex3fv(self.vertices[f[1]])
            glVertex3fv(self.vertices[f[2]])
        glEnd()

model = Model("simplescene.obj")

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
    
    #Draw stuff and move camera
    try :
        for (x,y,w,h) in faces :
            nx = x*aspectW
            ny = y*aspectH
            sx = (nx-WIDTH/2)/100
            sy = (ny-HEIGHT/2)/100
            if drawFace == 1 :
                #Normal way of drawing facelocations
                glBegin(GL_QUADS)
                glColor3f(1,0,0)
                glVertex3f(sx-0.1,sy-0.1,-0.1)
                glVertex3f(sx-0.1,sy+0.1,-0.1)
                glVertex3f(sx+0.1,sy+0.1,-0.1)
                glVertex3f(sx+0.1,sy-0.1,-0.1)
                glEnd()
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity() #Reset matrix
            glTranslatef(camx,camy,camz)
            gluLookAt(camx,camy,camz,-sx/10 + camx, -sy/10 + camy,-1 + camz,0,1,0)
            glMatrixMode(GL_MODELVIEW)
        model.draw()
    except Exception as e :
        print(e)

    #Handle Events
    for event in pygame.event.get() :
        if event.type == pygame.QUIT :
            pygame.quit()
            video_capture.release()
            cv2.destroyAllWindows()
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
