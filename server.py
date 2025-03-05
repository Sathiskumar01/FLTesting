from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from collections import deque

app = Flask(__name__)

# Load and preprocess dataset function
def load_and_preprocess_dataset(file_path):
    dataset = pd.read_csv(file_path)
    if 'ModulationType' in dataset.columns:
        dataset.drop(columns=['ModulationType'], inplace=True)
    dataset = dataset.apply(pd.to_numeric, errors='coerce')
    dataset.fillna(0, inplace=True)
    return dataset

# QoS Calculation
def calculate_qos(end_to_end_delay, packet_delivery_rate, tau_t=90, w_d=0.5, w_p=0.5, pdr_threshold=0.8):
    if end_to_end_delay < tau_t and packet_delivery_rate > pdr_threshold:
        return w_d * (tau_t - end_to_end_delay) / tau_t + w_p * (packet_delivery_rate - pdr_threshold) / pdr_threshold
    return 0

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.array([state], dtype=np.float32)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

agent = DQNAgent(4, 4)  # 4 states, 4 actions

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    file_path = data.get("file_path")

    dataset = load_and_preprocess_dataset(file_path)
    
    # Process QoS
    dataset['QoSScore'] = dataset.apply(lambda row: calculate_qos(row['EndToEndDelay'], row['PacketDeliveryRate']), axis=1)
    
    results = {"qos_scores": dataset['QoSScore'].tolist()}
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
