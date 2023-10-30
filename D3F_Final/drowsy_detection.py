import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
import time
#import datetime
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates


def get_mediapipe_app(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    """Initialize and return Mediapipe FaceMesh Solution Graph object"""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return face_mesh


def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Eye Aspect Ratio for one eye.

    Args:
        landmarks: (list) Detected landmarks list
        refer_idxs: (list) Index positions of the chosen landmarks
                            in order P1, P2, P3, P4, P5, P6
        frame_width: (int) Width of captured frame
        frame_height: (int) Height of captured frame

    Returns:
        ear: (float) Eye aspect ratio
    """
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points

def get_mar(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Mouth Aspect Ratio for the mouth.

    Args:
        landmarks: (list) Detected landmarks list
        refer_idxs: (list) Index positions of the chosen landmarks
                            in order P1, P2, P3, P4, P5, P6
        frame_width: (int) Width of captured frame
        frame_height: (int) Height of captured frame

    Returns:
        mar: (float) Mouth aspect ratio
    """
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # Mouth landmark (x, y)-coordinates
        P1_P2 = distance(coords_points[0], coords_points[1])
        P3_P4 = distance(coords_points[2], coords_points[3])
        P5_P6 = distance(coords_points[4], coords_points[5])
        P7_P8 = distance(coords_points[6], coords_points[7])


        # Compute the mouth aspect ratio
        mar = (P3_P4 + P5_P6 + P7_P8)/(3*P1_P2)

    except:
        mar = 0.0
        coords_points = None

    return mar, coords_points


def calculate_ear_mar(landmarks, left_eye_idxs, right_eye_idxs, mouth_idxs, image_w, image_h):
    # Calculate Eye aspect ratio
    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    Avg_EAR = (left_ear + right_ear) / 2.0

    # Calculate Mouth aspect ratio
    cur_MAR, mouth_lm_coordinates = get_mar(landmarks, mouth_idxs, image_w, image_h)

    return Avg_EAR, cur_MAR, (left_lm_coordinates, right_lm_coordinates, mouth_lm_coordinates)


def plot_landmarks(frame, left_lm_coordinates, right_lm_coordinates, mouth_lm_coordinates, color):
    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates, mouth_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(frame, coord, 2, color, -1)

    frame = cv2.flip(frame, 1)
    return frame

def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image


class VideoFrameHandler:
    def __init__(self):
        """
        Initialize the necessary constants, mediapipe app
        and tracker variables
        """
        # Left and right eye chosen landmarks.
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }

        # Mouth chosen landmarks.
        self.mouth_idxs ={ "mouth": [61, 291, 39, 181, 0, 17, 269, 405]}

        # Used for coloring landmark points.
        # Its value depends on the current EAR value.
        self.RED = (0, 0, 255)  # BGR
        self.GREEN = (0, 255, 0)  # BGR

        # Initializing Mediapipe FaceMesh solution pipeline
        self.facemesh_model = get_mediapipe_app()

        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
            "COLOR": self.GREEN,
            "play_alarm": False
        }

        self.EAR_txt_pos = (10, 30)
        self.MAR_txt_pos = (10, 60)
        self.eye_shut_counter = 0
        self.yawn_counter = 0
        self.alarm_counter = 0
        self.yawn_flag = False
        self.eye_shut_flag = False
        self.alarm_flag = False
        self.row_dict = {}

    def process(self, frame: np.array, thresholds: dict):
        """
        This function is used to implement our Drowsy detection algorithm

        Args:
            frame: (np.array) Input frame matrix.
            thresholds: (dict) Contains the two threshold values
                               WAIT_TIME and EAR_THRESH.

        Returns:
            The processed frame and a boolean flag to
            indicate if the alarm should be played or not.
        """

        # To improve performance,
        # mark the frame as not writeable to pass by reference.
        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape

        DROWSY_TIME_txt_pos = (10, int(frame_h // 2 * 1.7))
        ALM_txt_pos = (10, int(frame_h // 2 * 1.85))

        results = self.facemesh_model.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            EAR, MAR, coordinates = calculate_ear_mar(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], self.mouth_idxs["mouth"], frame_w, frame_h)
            frame = plot_landmarks(frame, coordinates[0], coordinates[1], coordinates[2], self.state_tracker["COLOR"])

            if MAR > thresholds["MAR_THRESH"]:
                if self.yawn_flag != True:
                    self.yawn_counter += 1
                    self.yawn_flag = True
            else:
                self.yawn_flag = False

            if EAR < thresholds["EAR_THRESH"]:

                # Increase DROWSY_TIME to track the time period with EAR less than the threshold
                # and reset the start_time for the next iteration.
                end_time = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED
                
                if self.eye_shut_flag != True:
                    self.eye_shut_counter += 1
                    self.eye_shut_flag = True

                if self.state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_alarm"] = True
                    plot_text(frame, "WAKE UP! WAKE UP", ALM_txt_pos, self.state_tracker["COLOR"])

                    if self.alarm_flag != True:
                        self.alarm_counter += 1
                        self.alarm_flag = True
                else:
                    self.alarm_flag = False

            else:
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.GREEN
                self.state_tracker["play_alarm"] = False
                self.eye_shut_flag = False

            EAR_txt = f"EAR: {round(EAR, 2)}"
            MAR_txt = f"MAR: {round(MAR, 2)}"
            DROWSY_TIME_txt = f"DROWSY: {round(self.state_tracker['DROWSY_TIME'], 3)} Secs"
            plot_text(frame, EAR_txt, self.EAR_txt_pos, self.state_tracker["COLOR"])
            plot_text(frame, MAR_txt, self.MAR_txt_pos, self.state_tracker["COLOR"])
            plot_text(frame, DROWSY_TIME_txt, DROWSY_TIME_txt_pos, self.state_tracker["COLOR"])

            # Save the information to a pandas dataframe
            # current_time = datetime.datetime.now()
            current_time=time.time()
            alarm_on = self.state_tracker["play_alarm"]

            self.row_dict={"timestamp": current_time, "EAR": EAR, "MAR": MAR, "eye_shut_counter": self.eye_shut_counter, "yawn_counter": self.yawn_counter, "alarm_counter": self.alarm_counter , "alarm_on": alarm_on}


        else:
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["DROWSY_TIME"] = 0.0
            self.state_tracker["COLOR"] = self.GREEN
            self.state_tracker["play_alarm"] = False
            
            # Flip the frame horizontally for a selfie-view display.
            frame = cv2.flip(frame, 1)

        return frame, self.state_tracker["play_alarm"] 
