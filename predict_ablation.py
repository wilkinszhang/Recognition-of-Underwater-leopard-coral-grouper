from ultralytics import YOLO

# Load a model
model = YOLO('/home/whut4/zwj/ultralytics/runs/detect/brackish_YOLOv8-ablation/weights/best.pt')  # load a custom model

# Validate the model
# metrics = model.predict(source='/home/whut4/zwj/0426underwater/dataset2/test/images',save=True,save_txt=True,save_conf=True)  # no arguments needed, dataset and settings remembered
metrics = model.predict(source='/home/whut4/zwj/brackish/test/images',save=True,save_txt=True,save_conf=True,name='brackish_YOLOv8-ablation_test')  # no arguments needed, dataset and settings remembered

print("hello")
