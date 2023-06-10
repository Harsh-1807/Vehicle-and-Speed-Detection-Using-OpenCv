import cv2
import numpy as np
import vehicles
import time

cnt = 0

cap = cv2.VideoCapture("c.mp4")

# Get width and height of video
w = cap.get(3)
h = cap.get(4)
frameArea = h * w
areaTH = frameArea / 400

# Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Kernals
kernalOp = np.ones((3, 3), np.uint8)
kernalCl = np.ones((11, 11), np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX
cars = []
max_p_age = 5
pid = 1

# Speed variables
fps = 0
scale = 1  # Scaling factor for speed estimation

start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    frame_count += 1

    for i in cars:
        i.age_one()

    fgmask = fgbg.apply(frame)

    if ret == True:
        # Binarization
        ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        # Opening i.e First Erode then Dilate
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)

        # Closing i.e First Dilate then Erode
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.float32(kernalCl))

        # Find Contours
        countours0, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in countours0:
            area = cv2.contourArea(cnt)
            print(area)
            if area > areaTH:
                ####Tracking######
                m = cv2.moments(cnt)
                cx = int(m['m10'] / m['m00'])
                cy = int(m['m01'] / m['m00'])
                x, y, w, h = cv2.boundingRect(cnt)

                new = True
                for i in cars:
                    if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                        new = False
                        i.updateCoords(cx, cy)

                        if i.getState() == '1':
                            if i.getDir() == 'down' and i.getY() > down_limit:
                                i.setDone()
                            elif i.getDir() == 'up' and i.getY() < up_limit:
                                i.setDone()
                        if i.timedOut():
                            index = cars.index(i)
                            cars.pop(index)
                            del i

                if new == True:  # If nothing is detected, create new
                    p = vehicles.Car(pid, cx, cy, max_p_age)
                    cars.append(p)
                    pid += 1

                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Calculate speed
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                speed = fps * scale
                cv2.putText(frame, f"Speed: {speed:.2f} km/h", (x, y - 10), font, 0.3, (0, 250, 150), 1, cv2.LINE_AA)

                if speed > 50:
                    # Save the frame as an image
                    cv2.putText(frame, "OVERSPEED!!!!", (x, y - 30), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imwrite(f"speeding_car_{frame_count}.jpg", frame)

        cv2.imshow('EDI_VEHICLE_AND_SPEED_DETECTION', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
