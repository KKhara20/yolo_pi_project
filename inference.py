import cv2
import numpy as np
import onnxruntime as ort
from mqtt_client import MqttPublisher
from config import MODEL_PATH, CONFIDENCE_THRESHOLD, INPUT_SIZE

# ─────────────────────────────────────────────────────────

def preprocess(image):
    img = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # Add batch dim
    return img

def postprocess(outputs, orig_shape):
    detections = outputs[0]
    boxes = []

    for det in detections:
        conf = det[4]
        if conf < CONFIDENCE_THRESHOLD:
            continue

        cls_id = int(det[5])
        if cls_id != 0:  # Only person class
            continue

        x, y, w, h = det[0], det[1], det[2], det[3]
        x1 = int((x - w / 2) * orig_shape[1] / INPUT_SIZE)
        y1 = int((y - h / 2) * orig_shape[0] / INPUT_SIZE)
        x2 = int((x + w / 2) * orig_shape[1] / INPUT_SIZE)
        y2 = int((y + h / 2) * orig_shape[0] / INPUT_SIZE)

        boxes.append((x1, y1, x2, y2, conf))

    return boxes

def main():
    cap = cv2.VideoCapture(0)
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    mqtt_pub = MqttPublisher()
    track_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        outputs = session.run(None, {input_name: input_tensor})
        boxes = postprocess(outputs, frame.shape)

        for x1, y1, x2, y2, conf in boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Send coordinates as JSON via MQTT
            mqtt_pub.publish_detection(track_id, cx, cy)

            # Draw the detection on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"person {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            track_id += 1

        cv2.imshow("YOLOv8 Person Detection", frame)
        if cv2.waitKey(1) == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
