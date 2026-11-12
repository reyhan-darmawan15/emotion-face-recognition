from ultralytics import YOLO

if __name__ == "__main__":
    # Load trained weights
    model = YOLO("./runs/classify/train/weights/best.pt")
    print("Trained weight loaded.")

    # Directory to put images to be predicted by the model
    # Change it by your own
    source = "./datasets/train/angry"

    # Run the prediction
    results = model(source, device=0)
    results.to_csv()
    print("Prediction finished.")