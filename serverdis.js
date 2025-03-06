const express = require("express");
const cors = require("cors");
const fs = require("fs");
const csv = require("csv-parser");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const app = express();

app.use(cors());
app.use(express.json());
const upload = multer({ dest: "uploads/" });

function loadAndPreprocessDataset(filePath) {
    return new Promise((resolve, reject) => {
        let dataset = [];
        fs.createReadStream(filePath)
            .pipe(csv())
            .on("data", (row) => {
                delete row["ModulationType"]; // Remove ModulationType if it exists
                for (let key in row) {
                    row[key] = parseFloat(row[key]) || 0; // Convert to float
                }
                dataset.push(row);
            })
            .on("end", () => resolve(dataset))
            .on("error", (error) => reject(error));
    });
}

function calculateSINR(Pk, hk, Ik, noisePower = 1e-9) {
    return (Pk * hk) / (Ik + noisePower);
}

function calculateQoSSINR(sinr, tauT = 90, wD = 0.5, wP = 0.5, sinrThreshold = 10) {
    if (sinr > sinrThreshold) {
        return wD * (tauT - sinr) / tauT + wP * (sinr - sinrThreshold) / sinrThreshold;
    }
    return 0;
}

class LocalDNN {
    constructor(inputSize, outputSize) {
        this.model = tf.sequential();
        this.model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [inputSize] }));
        this.model.add(tf.layers.dense({ units: outputSize, activation: 'linear' }));
        this.optimizer = tf.train.adam(0.001);
    }

    async predict(state) {
        const tensorState = tf.tensor2d([state], [1, state.length]);
        return this.model.predict(tensorState).data();
    }

    async train(state, target) {
        const tensorState = tf.tensor2d([state], [1, state.length]);
        const tensorTarget = tf.tensor2d([target], [1, target.length]);
        await this.model.fit(tensorState, tensorTarget, { epochs: 1 });
    }
}

class GlobalDNN {
    constructor(inputSize, outputSize) {
        this.model = tf.sequential();
        this.model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [inputSize] }));
        this.model.add(tf.layers.dense({ units: outputSize, activation: 'linear' }));
    }

    aggregateWeights(localModels) {
        const newWeights = localModels[0].model.getWeights().map((weights, i) => {
            const avgWeights = localModels.map(model => model.model.getWeights()[i]);
            return tf.mean(tf.stack(avgWeights), 0).arraySync();
        });
        this.model.setWeights(newWeights);
    }
}

class DistributedDQN {
    constructor(stateSize, actionSize) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.memory = [];
        this.gamma = 0.95;
        this.epsilon = 1.0;
        this.epsilonMin = 0.01;
        this.epsilonDecay = 0.995;
        this.model = this.buildModel();
    }

    buildModel() {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [this.stateSize] }));
        model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
        model.add(tf.layers.dense({ units: this.actionSize, activation: 'linear' }));
        model.compile({ loss: 'meanSquaredError', optimizer: tf.train.adam(0.001) });
        return model;
    }

    remember(state, action, reward, nextState, done) {
        this.memory.push({ state, action, reward, nextState, done });
    }

    async act(state) {
        if (Math.random() <= this.epsilon) {
            return Math.floor(Math.random() * this.actionSize);
        }
        const predictions = await this.model.predict(tf.tensor2d([state], [1, this.stateSize])).data();
        return predictions.indexOf(Math.max(...predictions));
    }

    async replay(batchSize) {
        if (this.memory.length < batchSize) return;
        const minibatch = this.memory.slice(-batchSize);
        for (const { state, action, reward, nextState, done } of minibatch) {
            const target = done ? reward : reward + this.gamma * Math.max(...(await this.model.predict(tf.tensor2d([nextState], [1, this.stateSize])).data()));
            const targetF = await this.model.predict(tf.tensor2d([state], [1, this.stateSize])).data();
            targetF[action] = target;
            await this.model.fit(tf.tensor2d([state], [1, this.stateSize]), tf.tensor2d([targetF], [1, targetF.length]), { epochs: 1 });
        }
        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
        }
    }
}

app.post("/upload", upload.single("file"), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "File is required" });
        }
        const dataset = await loadAndPreprocessDataset(req.file.path);
        const stateSize = 4; // Adjust based on your dataset
        const actionSize = 4; // Adjust based on your application
        const numVehicles = 5;
        const numEpisodes = 5;
        const M = 5;

        const globalModel = new GlobalDNN(stateSize, actionSize);
        const localModels = Array.from({ length: numVehicles }, () => new LocalDNN(stateSize, actionSize));
        const dqnAgents = Array.from({ length: numVehicles },)
        const dqnAgents = Array.from({ length: numVehicles }, () => new DistributedDQN(stateSize, actionSize));
        const urbanMetrics = { packet_delivery_rate: [], end_to_end_delay: [], task_satisfaction_rate: [], qos: [] };

        for (let episode = 0; episode < numEpisodes; episode++) {
            if (episode % M === 0) {
                console.log(`\nEpisode ${episode}: Global Model Aggregation`);
                globalModel.aggregateWeights(localModels);
            }

            for (let i = 0; i < numVehicles; i++) {
                const vehicleData = dataset.slice(0, Math.floor(dataset.length * 0.5)); // Sample half of the dataset
                for (const row of vehicleData) {
                    const state = [row.TransmissionPower, row.CurrentChannelPowerGain, row.CrossChannelPowerGain, row.QoSScore];
                    const action = await dqnAgents[i].act(state);
                    const sinr = calculateSINR(row.TransmissionPower, row.CurrentChannelPowerGain, row.CrossChannelPowerGain);
                    const reward = calculateQoSSINR(sinr);
                    const nextState = state; // In this case, we assume the next state is the same as the current state
                    const done = vehicleData.indexOf(row) === vehicleData.length - 1; // Mark the last row as done

                    dqnAgents[i].remember(state, action, reward, nextState, done);

                    // Store Metrics
                    urbanMetrics.packet_delivery_rate.push(row.PacketDeliveryRate);
                    urbanMetrics.end_to_end_delay.push(row.EndToEndDelay);
                    urbanMetrics.task_satisfaction_rate.push(row.TaskSatisfactionRate);
                    urbanMetrics.qos.push(row.QoSScore);
                }
                await dqnAgents[i].replay(16); // Train the agent with a batch size of 16
            }
        }

        // Return the QoS scores as a response
        res.json({ qos_scores: urbanMetrics.qos });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
