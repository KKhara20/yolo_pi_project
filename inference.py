import onnxruntime as ort
import numpy as np
import cv2
import time
import json
import paho.mqtt.client as mqtt
from config import MODEL_PATH

# ------------------ Configuration ------------------ #

INPUT_SIZE = (640,640)
CONF_THRESH = 0.3
IOU_THRESH = 0.45
MQTT_BROKER = "192.168.1.102"  # Replace with your IP
MQTT_PORT = 1883
MQTT_TOPIC = "pi/detections"

# ------------------ Utils ------------------ #
def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    return inter / np.maximum(union, 1e-6)

def fast_nms(boxes, scores, iou_threshold):
    if boxes.shape[0] == 0:
        return []
    indices = np.argsort(scores)[::-1]
    keep = []
    while indices.size > 0:
        current = indices[0]
        keep.append(current)
        if indices.size == 1:
            break
        ious = compute_iou(boxes[current], boxes[indices[1:]])
        indices = indices[1:][ious < iou_threshold]
    return keep

# ------------------ MQTT Client ------------------ #
class MqttPublisher:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.connect(MQTT_BROKER, MQTT_PORT)
        self.client.loop_start()

    def publish(self, topic, payload):
        self.client.publish(topic, json.dumps(payload))

# ------------------ Pre & Post Processing ------------------ #
def preprocess(frame, input_shape=(320, 320)):
    h, w = frame.shape[:2]
    r = min(input_shape[0] / h, input_shape[1] / w)
    nw, nh = int(w * r), int(h * r)
    img = cv2.resize(frame, (nw, nh))
    canvas = np.full((input_shape[1], input_shape[0], 3), 114, dtype=np.uint8)
    dw, dh = (input_shape[0] - nw) // 2, (input_shape[1] - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = img
    img = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0), canvas

def postprocess(output, image_shape, conf_thresh=0.3, iou_thresh=0.45):
    predictions = np.squeeze(output[0]).T
    scores = np.max(predictions[:, 4:], axis=1)
    class_ids = np.argmax(predictions[:, 4:], axis=1)
    mask = (class_ids == 0) & (scores > conf_thresh)
    if not np.any(mask):
        return [], [], []
    boxes = xywh2xyxy(predictions[mask][:, :4])
    boxes = np.clip(boxes, 0, [image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
    scores = scores[mask]
    class_ids = class_ids[mask]
    keep = fast_nms(boxes, scores, iou_thresh)
    return boxes[keep], scores[keep], class_ids[keep]

# ------------------ Inference ------------------ #
def run_inference():
    # Setup ONNX session
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Camera & MQTT
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 10)
    mqtt = MqttPublisher()

    print("🚀 Inference running. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        img_input, processed_img = preprocess(frame, INPUT_SIZE)
        output = session.run(None, {input_name: img_input})
        boxes, scores, class_ids = postprocess(output, processed_img.shape, CONF_THRESH, IOU_THRESH)
        end = time.time()

        # Publish detections
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            mqtt.publish(MQTT_TOPIC, {"id": int(i), "x": int(cx), "y": int(cy)})
q

        print(f"📸 FPS: {1 / (end - start):.2f}, Persons: {len(boxes)}")

        # Optional display
        # for box in boxes.astype(int):
        #     cv2.rectangle(processed_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # cv2.imshow("YOLOv8 Inference", processed_img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()
    mqtt.client.loop_stop()
    mqtt.client.disconnect()

run_inference()

