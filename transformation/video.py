import cv2
import argparse as ap

# Create a command-line argument parser.
aps = ap.ArgumentParser()
aps.add_argument("-V", "--videoid", type=int, default=0, help="Camera ID: int")
args = vars(aps.parse_args())

# Retrieve the camera ID from command-line arguments or use 0 as a default
videoId = args["videoid"] or 0

# Create a VideoCapture object for the default camera (usually camera index 0)
cap = cv2.VideoCapture(videoId)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Manipulate your camera here...
    flipedframe = cv2.flip(frame, 1)

    # Resize both the captured frame and the flipped frame
    resized_frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
    resized_flipedframe = cv2.resize(flipedframe, (0, 0), fx=0.6, fy=0.6)

    # Display the captured frame
    cv2.imshow("Original Video Capture", resized_frame)
    cv2.imshow("Fliped Video Capture", resized_flipedframe)

    # Press 'q' to exit the video capture loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the VideoCapture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()