import math
import numpy as np
import cv2

cap_video = cv2.VideoCapture(0)
contours = {}
approx = []
scale = 2

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while cap_video.isOpened():
    ret, frame = cap_video.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(frame, 80, 240, 3)
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if area > 1000:
                approx = cv2.approxPolyDP(cnt, 0.011 * cv2.arcLength(cnt, True), True)
                cornerCount = len(approx)
                x, y, w, h = cv2.boundingRect(approx)

                roi = frame[y:y + h, x:x + w]

                # red_threshold = 50
                mean_color = cv2.mean(roi)[:3]
                if cornerCount == 3:
                    if mean_color[2] > 50:
                        shape = "YIELD"
                    else:
                        shape = "triangle"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv2.putText(frame, f"{shape}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 255), 1,
                                cv2.LINE_AA)

                elif cornerCount == 8:
                    if mean_color[2] > 50 and mean_color[1] < 150 and mean_color[0] < 150:
                        shape = "STOP"
                    else:
                        shape = "octagon"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv2.putText(frame, f"{shape}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 255), 1,
                                cv2.LINE_AA)

                else:
                    pass

                cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3)

        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 1048689:
            break

cap_video.release()
cv2.destroyAllWindows()
