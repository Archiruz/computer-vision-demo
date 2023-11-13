import cv2
import numpy as np

def real_time_display():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Resize the frame to a smaller size
        frame = cv2.resize(frame, (640, 360))

        # Konversi dari BGR ke Grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Menampilkan hanya channel biru
        blue_channel_frame = frame.copy()
        blue_channel_frame[:, :, 1] = 0
        blue_channel_frame[:, :, 2] = 0

        # Deteksi objek berwarna merah
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        result_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Tampilkan hasil operasi pada layar
        cv2.imshow('Grayscale Video', grayscale_frame)
        cv2.imshow('Blue Channel Video', blue_channel_frame)
        cv2.imshow('Color Detection Video', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_display()