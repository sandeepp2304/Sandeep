import face_recognition
import cv2
import imutils

# Open the input movie file
input_movie = cv2.VideoCapture("D://Opencv-POC//livevideo//elonmusk.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_movie = cv2.VideoWriter('D://Opencv-POC//livevideo//elonmusk_output.avi', fourcc, 29.97, (640, 360))
output_movie1 = cv2.VideoWriter('D://Opencv-POC//livevideo//elonmusk_output1.mp4', fourcc, 29.97, (640, 360))
# Load some sample pictures and learn how to recognize them.
Elonmusk_image = face_recognition.load_image_file("D://Opencv-POC//livevideo//Elonmusk.png")
Elonmusk_face_encoding = face_recognition.face_encodings(Elonmusk_image)[0]

Musk1_image = face_recognition.load_image_file("D://Opencv-POC//livevideo//Musk1.png")
Musk1_face_encoding = face_recognition.face_encodings(Musk1_image)[0]

known_faces = [
    Elonmusk_face_encoding,
    Musk1_face_encoding
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    #rgb_frame = imutils.resize(rgb_frame,800)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        print(face_encoding)
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        name = None
        if match[0]:
            name = "Elonmusk"
        elif match[1]:
            name = "Musk1"

        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)
    output_movie1.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()