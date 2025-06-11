# config.py

CLIENT_ID   = "pi_cam_1"
MQTT_BROKER = "192.168.1.100"
MQTT_PORT   = 1883
MQTT_TOPIC  = f"{CLIENT_ID}/detections"
MODEL_PATH = r"C:\Users\karan\PycharmProjects\Yolo_pi_project#\deploy_pi\model\yolov8n_person.onnx"
CONFIDENCE_THRESHOLD = 0.3
INPUT_SIZE = 640