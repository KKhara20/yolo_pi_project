import cv2
from ultralytics import YOLO

def main():
    # Load the NCNN-exported YOLOv11n model folder
    model = YOLO(r"C:\Users\karan\PycharmProjects\Yolo_pi_project#\deploy_pi\yolo11n_ncnn_model")  # Make sure this folder contains .param and .bin files

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot access camera")

    print("🎯 Running YOLO11n on camera via NCNN backend")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame)

        # Annotate frame with bounding boxes and labels
        annotated = results[0].plot()

        # Compute and display FPS
        timing = results[0].speed  # {'preprocess': ..., 'inference': ..., 'postprocess': ...}
        total_ms = sum(timing.values())
        fps = 1000 / total_ms if total_ms > 0 else 0
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display result
        cv2.imshow("YOLO11n NCNN Inference", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
