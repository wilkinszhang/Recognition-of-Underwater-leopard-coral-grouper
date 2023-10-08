from ultralytics import YOLO



import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch    
import cv2
import numpy as np
import requests
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image

COLORS = np.random.uniform(0, 255, size=(80, 3))
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image


img = np.array(Image.open("/home/whut4/zwj/0426underwater/dataset/test/images/VID_20230414_153532-0001.jpg"))
img = cv2.resize(img, (640, 640))
rgb_img = img.copy()
img = np.float32(img) / 255
transform = transforms.ToTensor()
tensor = transform(img).unsqueeze(0)

# Load a model
model = YOLO('/home/whut4/zwj/ultralytics/runs/detect/PLGAT/weights/best.pt')  # load a custom model
# model.eval()
# model.cpu()
# target_layers = [model.model.model.model[-2]]
target_layers =[model.model.model[-4]]

# results = model([rgb_img])
cam = EigenCAM(model, target_layers,task='od')
grayscale_cam = cam(rgb_img)[0, :, :]
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
# Create a PIL Image object from the array
image = Image.fromarray(cam_image)

# Save the image as a JPEG file
image.save('output3.jpg')