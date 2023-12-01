import math
import numpy as np
import cv2


cap_video = cv2.VideoCapture(0)
contours = {}
approx = []
scale = 2

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

def calculate_angle(pt1, pt2, pt0):
    vector1 = [pt1[0][0] - pt0[0][0], pt1[0][1] - pt0[0][1]]
    vector2 = [pt2[0][0] - pt0[0][0], pt2[0][1] - pt0[0][1]]

    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    angle = math.acos(dot_product / (magnitude1 * magnitude2 + 1e-10))
    return math.degrees(angle)

while(cap_video.isOpened()):
    ret, frame = cap_video.read()
    if ret==True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(frame,80,240,3)
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if area > 1000:
                approx = cv2.approxPolyDP(cnt, 0.011 * cv2.arcLength(cnt, True), True)
                cornerCount = len(approx)
                x, y, w, h = cv2.boundingRect(approx)
                if cornerCount == 3:
                    shape = "triangle"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv2.putText(frame, "Triangle", (x , y), cv2.FONT_HERSHEY_SIMPLEX,  0.85, (255, 0, 255),1 ,cv2.LINE_AA)
                elif cornerCount == 6:
                    shape = "hexagon"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv2.putText(frame, "Hexagon", (x , y ), cv2.FONT_HERSHEY_COMPLEX,  0.85, (255, 0, 255),1 ,cv2.LINE_AA)
                elif cornerCount == 8:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv2.putText(frame, "Octagon", (x , y ), cv2.FONT_HERSHEY_COMPLEX, 0.85, (255, 0, 255),1, cv2.LINE_AA)
                    shape = "octagon"
                elif cornerCount > 10:
                    aspRatio = w / float(h)
                    if 0.95 < aspRatio < 1.05:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                        cv2.putText(frame, "Circle", (x , y ), cv2.FONT_HERSHEY_COMPLEX,  0.85, (255, 0, 255),1, cv2.LINE_AA)
                        shape = "circle"
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                        cv2.putText(frame, "Oval", (x , y ), cv2.FONT_HERSHEY_COMPLEX,  0.85, (255, 0, 255),1, cv2.LINE_AA)
                        shape = "oval"
                else:
                    pass
                cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3)

        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 1048689:
            break

cap_video.release()
cv2.destroyAllWindows()