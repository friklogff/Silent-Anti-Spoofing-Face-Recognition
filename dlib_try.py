"""

NAME : try_7

USER : admin

DATE : 9/10/2023

PROJECT_NAME : new_live_face

CSDN : friklogff
"""
import random
import time
import cv2
import numpy as np
from retinaface import Retinaface
import dlib
from imutils import face_utils





class BlinkDetection:
    def __init__(self):
        self.ear = None
        self.status = None
        self.frame_counter = 0
        self.blink_counter = 0
        self.EAR_THRESHOLD = 0.2  # 眨眼的 EAR 阈值

    def eye_aspect_ratio(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect(self, landmarks):
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        EAR_left = self.eye_aspect_ratio(left_eye)
        EAR_right = self.eye_aspect_ratio(right_eye)

        self.ear = (EAR_left + EAR_right) / 2.0

        if self.ear < 0.21:
            self.frame_counter += 1
            self.status = "Blinking"
        else:
            if self.frame_counter >= 2:  # 改为2次算检测结束
                self.blink_counter += 1
                self.frame_counter = 0
            self.status = "Open"

        return self.blink_counter, self.status, self.ear


class MouthDetection:

    def __init__(self):
        self.mStart, self.mEnd = (48, 68)
        self.mouth_counter = 0
        self.MAR_THRESHOLD = 0.5
        self.mouth_open = False  # 嘴巴状态，初始为闭上

    def mouth_aspect_ratio(self, mouth):
        A = np.linalg.norm(mouth[2] - mouth[9])
        B = np.linalg.norm(mouth[4] - mouth[7])
        C = np.linalg.norm(mouth[0] - mouth[6])
        mar = (A + B) / (2.0 * C)
        return mar

    def detect(self, landmarks):
        mouth = landmarks[self.mStart:self.mEnd]
        mar = self.mouth_aspect_ratio(mouth)

        if mar > self.MAR_THRESHOLD:
            if not self.mouth_open:  # 从闭上到张开
                self.mouth_counter += 1
                self.mouth_open = True
        else:
            if self.mouth_open:  # 从张开到闭上
                self.mouth_open = False

        return self.mouth_counter


class HeadPoseDetection:
    def __init__(self):
        self.left_counter = 0
        self.right_counter = 0

        self.nod_threshold = 10
        self.low_threshold = -10
        self.head_status = "neutral"

    def calculate_head_pose(self, shape):
        x, y = zip(*shape)
        face_center = (int(np.mean(x)), int(np.mean(y)))
        left_eye_center = np.mean(shape[36:42], axis=0)
        right_eye_center = np.mean(shape[42:48], axis=0)
        dX = right_eye_center[0] - left_eye_center[0]
        dY = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(dY, dX))
        return angle

    def detect(self, shape):
        angle = self.calculate_head_pose(shape)

        if angle > self.nod_threshold:
            self.head_status = "left"
            self.left_counter += 1
            return self.head_status, self.left_counter

        elif angle < self.low_threshold:
            self.head_status = "right"
            self.right_counter += 1
            return self.head_status, self.right_counter
        else:
            self.head_status = "neutral"

            return self.head_status, 0


class FaceDetection:
    def __init__(self, video_path, video_save_path="", video_fps=25.0, use_camera=False):

        self.name = None
        self.mouth_flag = False
        self.head_flag = False
        self.blink_flag = False
        self.random_flag = random.randint(1, 3)
        if use_camera:
            self.capture = cv2.VideoCapture(0)
        else:
            self.capture = cv2.VideoCapture(video_path)
        self.video_save_path = video_save_path
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
        self.ref, frame = self.capture.read()

        if not self.ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
        self.fps = 0.0
        self.flag = 0
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        self.blink_detector = BlinkDetection()
        self.mouth_detector = MouthDetection()
        self.head_pose_detector = HeadPoseDetection()

        self.nod_threshold = 10
        self.low_threshold = -10
        self.head_status = "neutral"
        self.blink_counter = 0
        self.mouth_counter = 0
        self.head_counter = 0
        self.ear = None
        self.status = None
        self.retinaface = Retinaface()

    def detect_blink(self, frame, landmarks):
        self.blink_counter, self.status, self.ear = self.blink_detector.detect(landmarks)
        cv2.putText(frame, "Blinks: {}".format(self.blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(self.ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Eyes Status: {}".format(self.status), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                    2)
        return self.blink_counter

    def detect_mouth(self, frame, landmarks):
        self.mouth_counter = self.mouth_detector.detect(landmarks)
        cv2.putText(frame, "Mouth Count: {}".format(self.mouth_counter), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        return self.mouth_counter

    def detect_head_pose(self, frame, gray, face_rectangle):
        shape = self.predictor(gray, face_rectangle)
        shape = face_utils.shape_to_np(shape)
        self.head_status, self.head_counter = self.head_pose_detector.detect(shape)
        cv2.putText(frame, "Head Status: {}".format(self.head_status), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255),
                    2)
        cv2.putText(frame, "Head Count: {}".format(self.head_counter), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255),
                    2)
        return self.head_counter

    def process_frame(self):
        t1 = time.time()
        self.ref, self.frame = self.capture.read()
        if not self.ref:
            return None
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        if self.flag == 1:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            old_image, self.name = self.retinaface.live_detect_image(self.frame, self.flag)
            self.frame = np.array(old_image)
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
            self.fps = (self.fps + (1. / (time.time() - t1))) / 2
            # print("fps= %.2f" % (self.fps))
            self.frame = cv2.putText(self.frame, "fps= %.2f" % self.fps, (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif len(faces) != 0:
            largest_index = self._largest_face(faces)
            face_rectangle = faces[largest_index]
            landmarks = np.matrix([[p.x, p.y] for p in self.predictor(self.frame, face_rectangle).parts()])
            if self.random_flag == 1:
                # 调用眨眼检测函数
                self.detect_blink(self.frame, landmarks)
                if self.blink_counter > 3:
                    self.blink_flag = True
                    self.random_flag = random.randint(1, 3)

            if self.random_flag == 2:

                # 调用嘴巴动作检测函数
                self.detect_mouth(self.frame, landmarks)
                if self.mouth_counter > 3:
                    self.mouth_flag = True
                    self.random_flag = random.randint(1, 3)
            if self.random_flag == 3:
                # 调用头部姿势检测函数
                self.detect_head_pose(self.frame, gray, face_rectangle)
                if self.head_counter == 0:
                    self.head_flag = True
                    self.random_flag = random.randint(1, 3)
            if self.blink_flag and self.mouth_flag and self.head_flag:
                self.flag = 1

        if self.video_save_path != "":
            self.out.write(self.frame)

        return self.ref, self.frame

    def _largest_face(self, dets):
        if len(dets) == 1:
            return 0
        face_areas = [(det.right() - det.left()) * (det.bottom() - det.top()) for det in dets]
        largest_area = face_areas[0]
        largest_index = 0
        for index in range(1, len(dets)):
            if face_areas[index] > largest_area:
                largest_index = index
                largest_area = face_areas[index]
        print("largest_face index is {} in {} faces".format(largest_index, len(dets)))
        return largest_index

    def release(self):
        print("Video Detection Done!")
        self.capture.release()
        if self.video_save_path != "":
            print("Save processed video to the path:" + self.video_save_path)
            self.out.release()

    def get_blink_counter(self):
        return self.blink_counter

    def get_mouth_counter(self):
        return self.mouth_counter

    def get_head_counter(self):
        return self.head_counter

    def get_flag(self):
        return self.flag

    def get_name(self):
        return self.name



if __name__ == "__main__":
    detector = FaceDetection('R.mp4')  # 使用摄像头，也可以指定视频文件路径

    while True:
        flag = detector.get_flag()

        ref, frame = detector.process_frame()
        blink_counter = detector.get_blink_counter()
        mouth_counter = detector.get_mouth_counter()
        head_counter = detector.get_head_counter()

        #        print(blink_counter, mouth_counter, head_counter)
        #       if blink_counter > 9:
        #          break

        if frame is None:
            break
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if flag == 1:
            print(flag)
            cv2.imwrite("last_frame.png", frame)
            # print(fname)
            break
    detector.release()
    cv2.destroyAllWindows()
