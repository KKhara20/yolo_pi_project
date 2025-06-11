import numpy as np
import cv2

# Class ID 0 only: 'person'
class_names = ['person']
color = (0, 255, 0)  # Green for person

def compute_iou(box, boxes):
    """Compute IoU between a single box and an array of boxes."""
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return intersection / np.maximum(union, 1e-6)

def fast_nms(boxes, scores, iou_threshold):
    """Vectorized NMS for a single class."""
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

def person_nms(boxes, scores, class_ids, iou_threshold=0.45):
    """NMS for class_id == 0 (person) only."""
    mask = class_ids == 0
    if not np.any(mask):
        return []

    person_boxes = boxes[mask]
    person_scores = scores[mask]
    keep_indices = fast_nms(person_boxes, person_scores, iou_threshold)

    return np.where(mask)[0][keep_indices]  # return original indices

def xywh2xyxy(x):
    """Convert [x_center, y_center, w, h] to [x1, y1, x2, y2]."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def draw_detections(image, boxes, scores, class_ids):
    """Draw only 'person' class (id=0)."""
    image = image.copy()
    h, w = image.shape[:2]
    font_scale = min(h, w) * 0.0006
    thickness = int(min(h, w) * 0.001)

    mask = class_ids == 0
    if not np.any(mask):
        return image

    boxes = boxes[mask].astype(int)
    scores = scores[mask]

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        label = f'person {int(score * 100)}%'
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th - 4), color, -1)
        cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    return image
