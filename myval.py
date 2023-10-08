from ultralytics import YOLO

# Load a model
model = YOLO('/home/whut4/zwj/ultralytics/runs/detect/cfg2/weights/best.pt')  # load a custom model

# Validate the model
metrics = model.val(data='/home/whut4/zwj/0426underwater/dataset4/data.yaml')

metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category