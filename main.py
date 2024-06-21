import face_recognition
import cv2
from multiprocessing import Process, Manager, cpu_count, set_start_method
import time
import numpy as np
import threading
import platform
from PIL import Image
import io
# This is a little bit complicated (but fast) example of running face recognition on live video from your webcam.
# This example is using multiprocess.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
  'databaseURL':"https://face-recognition-fb3bc-default-rtdb.firebaseio.com/",
  'storageBucket':"face-recognition-fb3bc.appspot.com"
})
bucket = storage.bucket()

# Get next worker's id
def next_id(current_id, worker_num):
    if current_id == worker_num:
        return 1
    else:
        return current_id + 1

# Get previous worker's id
def prev_id(current_id, worker_num):
    if current_id == 1:
        return worker_num
    else:
        return current_id - 1

# A subprocess use to capture frames.
def capture(read_frame_list, Global, worker_num):
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    # video_capture.set(3, 640)  # Width of the frames in the video stream.
    # video_capture.set(4, 480)  # Height of the frames in the video stream.
    # video_capture.set(5, 30) # Frame rate.
    print("Width: %d, Height: %d, FPS: %d" % (video_capture.get(3), video_capture.get(4), video_capture.get(5)))

    while not Global.is_exit:
        # If it's time to read a frame
        if Global.buff_num != next_id(Global.read_num, worker_num):
            # Grab a single frame of video
            ret, frame = video_capture.read()
            read_frame_list[Global.buff_num] = frame
            Global.buff_num = next_id(Global.buff_num, worker_num)
        else:
            time.sleep(0.01)

    # Release webcam
    video_capture.release()

# Many subprocess use to process frames.
def process(worker_id, read_frame_list, write_frame_list, Global, worker_num):
    known_face_encodings = Global.known_face_encodings

    while not Global.is_exit:

        # Wait to read
        while Global.read_num != worker_id or Global.read_num != prev_id(Global.buff_num, worker_num):
            # If the user has requested to end the app, then stop waiting for webcam frames
            if Global.is_exit:
                break

            time.sleep(0.01)

        # Delay to make the video look smoother
        time.sleep(Global.frame_delay)

        # Read a single frame from frame list
        frame_process = read_frame_list[worker_id]

        # Expect next worker to read frame
        Global.read_num = next_id(Global.read_num, worker_num)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = np.ascontiguousarray(frame_process[:, :, ::-1])
        
        # Find all the faces and face encodings in the frame of video, cost most time
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
              first_match_index = matches.index(True)
              cv2.rectangle(frame_process, (left, top), (right, bottom), (0, 255, 0), 2)
            else:
            # Draw a box around the face
              cv2.rectangle(frame_process, (left, top), (right, bottom), (0, 0, 255), 2)

        # Wait to write
        while Global.write_num != worker_id:
            time.sleep(0.01)

        # Send frame to global
        write_frame_list[worker_id] = frame_process

        # Expect next worker to write frame
        Global.write_num = next_id(Global.write_num, worker_num)


if __name__ == '__main__':

    # Fix Bug on MacOS
    if platform.system() == 'Darwin':
        set_start_method('forkserver')

    # Global variables
    Global = Manager().Namespace()
    Global.buff_num = 1
    Global.read_num = 1
    Global.write_num = 1
    Global.frame_delay = 0
    Global.is_exit = False
    read_frame_list = Manager().dict()
    write_frame_list = Manager().dict()

    # Number of workers (subprocess use to process frames)
    worker_num = max(2, cpu_count() - 1)

    # Subprocess list
    p = []

    # Create a thread to capture frames (if uses subprocess, it will crash on Mac)
    p.append(threading.Thread(target=capture, args=(read_frame_list, Global, worker_num,)))
    p[0].start()

    ## Getting the files in the firebase storage
    def files_in_folder(folder):
      files = bucket.list_blobs(prefix=folder)
      return [blob.name for blob in files if not blob.name.endswith('/')]
    
    ## Getting the image in integers to the encoding function
    def getting_image(file_name):
      blob = bucket.blob(file_name)
      image_bytes = blob.download_as_bytes()
      return Image.open(io.BytesIO(image_bytes))

    ## function to fixed a color code error
    def to_numpy(image):
      if image.mode != 'RGB':
        image = image.convert('RGB')
      return np.array(image)
    
    ## To encode the images
    def encoding_images(images):
        encoded_images = []
        for image in images:
          encoded_image = face_recognition.face_encodings(image)[0]
          encoded_images.append(encoded_image)
        return encoded_images
    
    ## All the functions per image in the storage of firebase
    def importing_images():
        folder = 'images'
        fileNames = files_in_folder(folder)
        images = []
        for file in fileNames:
          image = getting_image(file)
          image_np = to_numpy(image)
          images.append(image_np)
        return images
    
    # Getting the images
    images = importing_images()

    # Create arrays of known face encodings and their names
    Global.known_face_encodings = encoding_images(images)

    print("Encoding and importing successfull")

    # Create workers
    for worker_id in range(1, worker_num + 1):
        p.append(Process(target=process, args=(worker_id, read_frame_list, write_frame_list, Global, worker_num,)))
        p[worker_id].start()

    # Start to show video
    last_num = 1
    fps_list = []
    tmp_time = time.time()
    while not Global.is_exit:
        while Global.write_num != last_num:
            last_num = int(Global.write_num)

            # Calculate fps
            delay = time.time() - tmp_time
            tmp_time = time.time()
            fps_list.append(delay)
            if len(fps_list) > 5 * worker_num:
                fps_list.pop(0)
            fps = len(fps_list) / np.sum(fps_list)
            print("fps: %.2f" % fps)

            # Calculate frame delay, in order to make the video look smoother.
            # When fps is higher, should use a smaller ratio, or fps will be limited in a lower value.
            # Larger ratio can make the video look smoother, but fps will hard to become higher.
            # Smaller ratio can make fps higher, but the video looks not too smoother.
            # The ratios below are tested many times.
            if fps < 6:
                Global.frame_delay = (1 / fps) * 0.75
            elif fps < 20:
                Global.frame_delay = (1 / fps) * 0.5
            elif fps < 30:
                Global.frame_delay = (1 / fps) * 0.25
            else:
                Global.frame_delay = 0

            # Display the resulting image
            cv2.imshow('Video', write_frame_list[prev_id(Global.write_num, worker_num)])

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            Global.is_exit = True
            break

        time.sleep(0.01)

    # Quit
    cv2.destroyAllWindows()