import os
import shutil
import warnings
# import cv2
import io
 
from PIL import Image
warnings.filterwarnings("error", category=UserWarning)
 
base_dir = "./dataset"

def is_read_successfully(file):
    try:
        imgFile = Image.open(file)
        return True
    except Exception:
        return False


i = 0
for r in os.listdir(base_dir):
    if r=='.DS_Store':#在这里我们在 .DS_Store 跳过,如果要是有需要的删除.DS_Store文件，可以进行微调
        print(base_dir,r)
        continue 
    print(r)
    for j  in os.listdir(os.path.join('./dataset',r)):
        if not is_read_successfully(f"{base_dir}/{r}/{j}"):
            i = i+1
            print(i)

