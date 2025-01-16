import requests
from PIL import Image
import numpy as np

# 실제 이미지 경로와 원하는 instruction으로 변경하세요
image_path = "/home/work/open_x_dataset/LAPA/imgs/bridge_inference.jpg"
image = Image.open(image_path)
image = np.array(image)
instruction = "Move the robot arm to pick up the object"

# 서버에 요청 보내기
response = requests.post("http://localhost:32820/act", json={
    "image": image.tolist(),
    "instruction": instruction
})

# 응답 받기
predicted_action = response.json()
print("예측된 액션:", predicted_action)