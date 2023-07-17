import face_recognition
import os
import cv2

KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.5 #The lower the tolerance, the more "strict" the labels will be
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  #'hog', other one can be 'cnn' 

# video=cv2.VideoCapture("try.mp4") # Also can put a video Name
video=cv2.VideoCapture(0) # Also can put a video Name

print('Loading known faces...')
known_faces = []
known_names = []

# Load known faces and their names
for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        encodings = face_recognition.face_encodings(image)
        for encoding in encodings:  #can handle images with multiple faces
            known_faces.append(encoding)
            known_names.append(name) 
        # encoding = pickle.load(open(f"{name}/{filename}","rb"))
        known_faces.append(encoding)
        known_names.append(name)

print('Processing unknown faces...')
counter = 0
skip_frames = 3  # Process every 5th frame

while True:
    ret, image = video.read()
    if not ret:
        break
    
    counter += 1
    if counter % skip_frames != 0:
        continue

    # Resize the frame to a smaller size for faster processing
    image = cv2.resize(image, (640, int(image.shape[0] * 640 / image.shape[1])))
    # we can do in this way also resize 

    #image = cv2.resize(image, (1200, 640))

    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    
    for face_encoding, face_location in zip(encodings, locations):

        # We use compare_faces (but might use face_distance as well)
        # Returns array of True/False values in order of passed known_faces
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

        # Since order is being preserved, we check if any face was found then grab index
        # then label (name) of first matching known face withing a tolerance
        match = 'Unknown'
        if True in results:  # If at least one is true, get a name of first of found labels
            match = known_names[results.index(True)]
            # print(f'match found - {match} ')


        # Each location contains positions in order: top, right, bottom, left
        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])
        color = [0,255,0]
        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)# Paint frame
        # Now we need smaller, filled grame below for a name
        # This time we use bottom in both corners - to start from bottom and move 50 pixels down
        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2] + 22)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), FONT_THICKNESS)
    
    cv2.imshow("Frame", image)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cv2.destroyAllWindows()
video.release()