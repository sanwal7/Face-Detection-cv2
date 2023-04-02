import cv2

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize variables for face tracking
prev_faces = []
cur_faces = []

# Create a VideoCapture object to read from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

    # Update the list of current faces
    cur_faces = []

    # Iterate through each detected face
    for (x,y,w,h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        # Add the current face to the list of current faces
        cur_faces.append((x,y,w,h))

    # If there are previous faces, check for movement
    if prev_faces:
        # Iterate through each previous face
        for (px,py,pw,ph) in prev_faces:
            # Find the closest current face to the previous face
            closest_face = None
            closest_distance = float('inf')

            for (cx,cy,cw,ch) in cur_faces:
                distance = ((px-cx)**2 + (py-cy)**2)**0.5

                if distance < closest_distance:
                    closest_distance = distance
                    closest_face = (cx,cy,cw,ch)

            # If the closest face is close enough to the previous face, assume it is the same face
            if closest_face and closest_distance < 50:
                # Draw a rectangle around the face
                cv2.rectangle(frame,(closest_face[0],closest_face[1]),(closest_face[0]+closest_face[2],closest_face[1]+closest_face[3]),(255,0,0),2)

                # Add the current face to the list of current faces
                cur_faces.append((closest_face[0],closest_face[1],closest_face[2],closest_face[3]))

    # Update the list of previous faces
    prev_faces = cur_faces

    # Show the frame with face detection
    cv2.imshow('Face Detection', frame)

    # Wait for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()