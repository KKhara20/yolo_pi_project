# config.py

CLIENT_ID   = "pi_cam_1"
MQTT_BROKER = "192.168.1.102"
MQTT_PORT   = 1883
MQTT_TOPIC  = f"{CLIENT_ID}/detections"
MODEL_PATH = r"/home/user/yolo_pi_project/model/yolov8n_person.onnx"
CONFIDENCE_THRESHOLD = 0.3
INPUT_SIZE = 640
