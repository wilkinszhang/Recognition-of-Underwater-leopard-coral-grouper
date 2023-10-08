from ultralytics import YOLO

def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 4
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
    for k, v in model.named_parameters(): 
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze): 
            print(f'freezing {k}') 
            v.requires_grad = False 
    print(f"{num_freeze} layers are freezed.")

if __name__ == '__main__':
    # 加载模型
    model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
    results = model.train(data="/home/whut4/zwj/brackish/data.yaml", epochs=20, batch=64, workers=8, close_mosaic=0, name='brackish_YOLOv8-ablation')  # 训练模型