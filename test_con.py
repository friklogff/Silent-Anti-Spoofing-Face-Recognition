import os
import cv2
import imutils
import numpy as np
import argparse
import warnings
import time

from imutils.video import VideoStream
predictions = None

from retinaface import Retinaface
from src.anti_spoof_predict_pro import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')


def check_image(image):
    height, width, channel = image.shape
    if width / height != 3 / 4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def detect_image(img, temp_img_path):
    retinaface = Retinaface()

    image = cv2.imread(img)
    if image is None:
        print('Open Error! Try again!')
        return
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r_image,name = retinaface.detect_image(image)
        r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Processed Image", r_image)
        cv2.waitKey(0)
        if temp_img_path != "":
            cv2.imwrite(temp_img_path, r_image)
            print("Save processed img to the path: " + temp_img_path)
            print(name)
            return temp_img_path


def load(model_dir, device_id, image_path = 'captured_photo.jpg'):
    global prediction
    model_test = AntiSpoofPredict(device_id, model_dir)
    image_cropper = CropImage()
    frame = cv2.imread(image_path)  # 从拍摄的照片文件中读取图像
    image_bbox = model_test.get_bbox(frame)
    prediction = np.zeros((1, 3))
    test_speed = 0
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()  # 记录预测开始时间
        # 使用模型进行预测
        predictions = model_test.predict_batch([img])
        # 将当前模型的预测结果添加到总预测中
        prediction += predictions[model_name]
        test_speed += time.time() - start  # 计算预测所花时间
        break


def test_and_detect(model_dir, device_id, image_path):
    global prediction
    model_test = AntiSpoofPredict(device_id, model_dir)
    image_cropper = CropImage()

    frame = cv2.imread(image_path)  # 从拍摄的照片文件中读取图像

    image_bbox = model_test.get_bbox(frame)
    prediction = np.zeros((1, 3))
    test_speed = 0
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()  # 记录预测开始时间

        # 使用模型进行预测
        predictions = model_test.predict_batch([img])

        # 将当前模型的预测结果添加到总预测中
        prediction += predictions[model_name]
        test_speed += time.time() - start  # 计算预测所花时间

    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    if label == 1:
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)
    print("Prediction cost {:.2f} s".format(test_speed))  # 打印预测所花时间

    cv2.rectangle(
        frame,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        frame,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5 * frame.shape[0] / 1024, color)

    cv2.imshow('frame', frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    # 调用detect_image函数处理拍摄的照片
    temp_img_path = "processed_photo.jpg"
    detect_image(image_path, temp_img_path)


def take_photo(camera_index, temp_img_path="temp.jpg"):
    # 打开摄像头
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("无法获取图像帧")
            break

        cv2.imshow("拍照", frame)

        key = cv2.waitKey(1)
        if key == 32:  # 按下空格键退出
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        face = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
        if len(face) > 0:  # 检查是否检测到人脸
            # 检测到人脸，立即拍照
            cv2.imwrite(temp_img_path, frame)
            print("已拍照并保存为：" + temp_img_path)
            break

    # 释放摄像头和关闭窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    args = parser.parse_args()
    load(args.model_dir, args.device_id)
    # 调用拍照函数并传递给test_and_detect
    temp_img_path = "captured_photo.jpg"
    take_photo(0,temp_img_path=temp_img_path)
    test_and_detect(args.model_dir, args.device_id, temp_img_path)
