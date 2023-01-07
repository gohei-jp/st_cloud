import cv2
import streamlit as st
from PIL import Image
import time
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

from huesdk import Hue
brige = {"id":"001788fffeb14d9c","ip":"192.168.40.149","port":443}
username = 'YmUqdKYQe7ud7QGeA3vCKJTo3z4jdSoxeDuO97kn'
hue = Hue(bridge_ip=brige.get('ip'), username=username)
light = hue.get_light(id_=10)
light.on()

def distance(a, b):
    return np.linalg.norm(np.array(a)-np.array(b))

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# file='rtsp://192.168.40.108/:554/user=admin&password=&channel=1&stream=0.sdp?'
cap = cv2.VideoCapture(0)

loop_time_sec = 0
turn_off_light_timer = 0
turn_on_light_timer = 0
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  image_loc = st.empty()
  while cap.isOpened():
    loop_start = time.perf_counter()

    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
    #   for face_landmarks in results.multi_face_landmarks:
    #     mp_drawing.draw_landmarks(
    #         image=image,
    #         landmark_list=face_landmarks,
    #         connections=mp_face_mesh.FACEMESH_TESSELATION,
    #         landmark_drawing_spec=None,
    #         connection_drawing_spec=mp_drawing_styles
    #         .get_default_face_mesh_tesselation_style())

        for face_landmarks in results.multi_face_landmarks:
            marks = face_landmarks.landmark
            base_distance = distance((marks[144].x,marks[144].y),
                                    (marks[145].x,marks[145].y))
            left_eye_distance = distance((marks[374].x,marks[374].y),
                                    (marks[386].x,marks[386].y))
            right_eye_distance = distance((marks[145].x,marks[145].y),
                                    (marks[159].x,marks[159].y))
            if base_distance > left_eye_distance and base_distance > right_eye_distance:
                turn_off_light_timer += loop_time_sec
                turn_on_light_timer = 0
            elif base_distance > left_eye_distance or base_distance > right_eye_distance:
                turn_on_light_timer += loop_time_sec
                turn_off_light_timer = 0
            else:
                turn_on_light_timer = 0
                turn_off_light_timer = 0

    if turn_on_light_timer > 0.5:
        light.set_brightness(254)
    elif turn_off_light_timer > 0.5:
        gradient = min(int(turn_off_light_timer*50), 180)
        light.set_brightness(light.bri-gradient)

    # Flip the image horizontally for a selfie-view display.
    # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_loc.image(image)
    loop_time_sec = time.perf_counter()-loop_start
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()