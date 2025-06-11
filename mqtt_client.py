# mqtt_client.py

import paho.mqtt.client as mqtt
import threading
import time
import json

# Config – update these as needed
MQTT_BROKER = "192.168.1.100"  # Replace with your laptop's IP
MQTT_PORT = 1883
MQTT_TOPIC = "pi/detections"
CLIENT_ID = "pi_client"

class MqttPublisher:
    def __init__(self):
        self.client = mqtt.Client(client_id=CLIENT_ID, protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.connected = False

        self._connect_thread = threading.Thread(target=self._connect_with_retries, daemon=True)
        self._connect_thread.start()

    def _connect_with_retries(self, retries=5, delay=3):
        for attempt in range(1, retries + 1):
            try:
                self.client.connect(MQTT_BROKER, MQTT_PORT)
                self.client.loop_start()
                print(f"[MQTT] Connection attempt {attempt} successful.")
                return
            except Exception as e:
                print(f"[MQTT] Attempt {attempt} failed: {e}")
                time.sleep(delay)
        print("[MQTT] Failed to connect after retries.")

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print("[MQTT] Connected to broker.")
        self.connected = True

    def on_disconnect(self, client, userdata, rc):
        print("[MQTT] Disconnected from broker.")
        self.connected = False

    def publish(self, topic: str, data: dict):
        if self.connected:
            try:
                self.client.publish(topic, json.dumps(data))
            except Exception as e:
                print(f"[MQTT] Publish failed: {e}")
        else:
            print("[MQTT] Not connected. Skipping publish.")
