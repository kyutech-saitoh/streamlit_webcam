import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

def process(image, is_image, is_landmarks):
    out_image = image.copy()

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:

        all_left_eye_idxs = list(mp.solutions.face_mesh.FACEMESH_LEFT_EYE)
        all_left_eye_idxs = set(np.ravel(all_left_eye_idxs)) 
        all_right_eye_idxs = list(mp.solutions.face_mesh.FACEMESH_RIGHT_EYE)
        all_right_eye_idxs = set(np.ravel(all_right_eye_idxs))       
        all_idxs = all_left_eye_idxs.union(all_right_eye_idxs)


        results = face_mesh.process(image)

        (image_height, image_width) = image.shape[:2]

        black_image = np.zeros((image_height, image_width, 3), np.uint8)
        white_image = black_image + 255

        if is_image == True:
            out_image = white_image.copy()
            
        if is_landmarks == True:
            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                   for landmark in face.landmark:               
                        x = int(landmark.x * image_width)
                        y = int(landmark.y * image_height)
                        cv2.circle(out_image, center=(x, y), radius=2, color=(0, 0, 255), thickness=-1)
                        cv2.circle(out_image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)    

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                for landmark_idx, landmark in enumerate(face):
                    if landmark_idx in all_idxs:
                        x = face.landmark[landmark_idx].x
                        y = face.landmark[landmark_idx].y
                        x = int(landmark.x * image_width)
                        y = int(landmark.y * image_height)
                        cv2.circle(out_image, center=(x, y), radius=2, color=(255, 0, 255), thickness=-1)
                        cv2.circle(out_image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)    
    
    return cv2.flip(out_image, 1)
    
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def __init__(self) -> None:
        self.is_image = True
        self.is_landmarks = True
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = process(img, self.is_image, self.is_landmarks)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.is_image = st.checkbox("show camera image", value=True)
    webrtc_ctx.video_processor.is_landmarks = st.checkbox("draw landmarks", value=True)
    
