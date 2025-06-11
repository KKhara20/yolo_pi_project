import cv2
import numpy as np
import onnxruntime
from utils import multiclass_nms, xywh2xyxy, draw_detections
from config import MODEL_PATH

CONF_THRESH = 0.3
IOU_THRESH = 0.45
INPUT_SIZE = (640, 640)

def preprocess_letterbox(frame, input_shape=(640, 640), pad_color=(114, 114, 114)):
    h, w = frame.shape[:2]
    r = min(input_shape[0] / h, input_shape[1] / w)
    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_shape[1], input_shape[0], 3), pad_color, dtype=np.uint8)
    dw = (input_shape[0] - new_w) // 2
    dh = (input_shape[1] - new_h) // 2
    canvas[dh:dh+new_h, dw:dw+new_w] = resized
    img = canvas.copy()
    img_input = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    return img_input, img

def postprocess(output, input_image_shape, conf_thresh=0.3, iou_thresh=0.45):
    predictions = np.squeeze(output[0]).T  # (8400, 84)
    scores = np.max(predictions[:, 4:], axis=1)
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Keep only person class (class 0)
    person_mask = (class_ids == 0) & (scores > conf_thresh)
    predictions = predictions[person_mask]
    scores = scores[person_mask]
    class_ids = class_ids[person_mask]

    if len(scores) == 0:
        return [], [], []

    boxes = predictions[:, :4]
    boxes = xywh2xyxy(boxes)
    boxes = np.clip(boxes, 0, [input_image_shape[1], input_image_shape[0], input_image_shape[1], input_image_shape[0]])

    indices = multiclass_nms(boxes, scores, class_ids, iou_thresh, class_filter=[0])
    return boxes[indices], scores[indices], class_ids[indices]

def main():
    session = onnxruntime.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    print("🚀 YOLOv8 Inference running. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_input, letterboxed_img = preprocess_letterbox(frame, INPUT_SIZE)
        outputs = session.run(None, {input_name: img_input})
        boxes, scores, class_ids = postprocess(outputs, letterboxed_img.shape, CONF_THRESH, IOU_THRESH)

        annotated = draw_detections(letterboxed_img, boxes, scores, class_ids)
        cv2.imshow("YOLOv8 - Person Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
