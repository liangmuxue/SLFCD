from ultralytics import YOLO

if __name__ == '__main__':
    # build from YAML and transfer weights
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')
    # Train the model
    model.train(data='./VOCData/mydata.yaml', epochs=300, device="cuda:1", workers=6)
        
    
# tensorboard --logdir runs/detect/train
