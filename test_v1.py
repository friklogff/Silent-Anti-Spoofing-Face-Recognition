# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True

def test(model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    cap = cv2.VideoCapture(0)  # 使用电脑摄像头获取实时视频流

    while True:
        ret, frame = cap.read()  # 读取一帧
        if not ret:
            break

        height, width, _ = frame.shape
        if width/height > 3/4:
            new_width = int(height * 3/4)
            start = (width - new_width) // 2
            frame = frame[:, start:start+new_width]
        elif width/height < 3/4:
            new_height = int(width * 4/3)
            start = (height - new_height) // 2
            frame = frame[start:start+new_height, :]

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
            start = time.time()
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
            test_speed += time.time()-start

        label = np.argmax(prediction)
        value = prediction[0][label]/2
        if label == 1:
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (255, 0, 0)
        else:
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
        print("Prediction cost {:.2f} s".format(test_speed))
        cv2.rectangle(
            frame,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            frame,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5*frame.shape[0]/1024, color)

        cv2.imshow('frame', frame)  # 显示结果
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
            break

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
    test(args.model_dir, args.device_id)
