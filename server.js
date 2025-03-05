const express = require("express");
const fs = require("fs");
const csv = require("csv-parser");
const tf = require("@tensorflow/tfjs-node");

const app = express();
app.use(express.json());

// Load and preprocess dataset
function loadAndPreprocessDataset(filePath) {
    return new Promise((resolve, reject) => {
        let dataset = [];
        fs.createReadStream(filePath)
            .pipe(csv())
            .on("data", (row) => {
                delete row["ModulationType"]; // Remove unnecessary column
                let processedRow = {};
                for (let key in row) {
                    processedRow[key] = parseFloat(row[key]) || 0;
                }
                dataset.push(processedRow);
            })
            .on("end", () => resolve(dataset))
            .on("error", (error) => reject(error));
    });
}

// QoS Calculation
function calculateQoS(endToEndDelay, packetDeliveryRate, tauT = 90, wD = 0.5, wP = 0.5, pdrThreshold = 0.8) {
    if (endToEndDelay < tauT && packetDeliveryRate > pdrThreshold) {
        return wD * (tauT - endToEndDelay) / tauT + wP * (packetDeliveryRate - pdrThreshold) / pdrThreshold;
    }
    return 0;
}

// DQN Agent Class
class DQNAgent {
    constructor(stateSize, actionSize) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.epsilon = 1.0;
        this.epsilonMin = 0.01;
        this.epsilonDecay = 0.995;
        this.model = this.buildModel();
    }

    buildModel() {
        return tf.sequential({
            layers: [
                tf.layers.dense({ units: 16, activation: "relu", inputShape: [this.stateSize] }),
                tf.layers.dense({ units: 16, activation: "relu" }),
                tf.layers.dense({ units: this.actionSize, activation: "linear" })
            ]
        });
    }

    async act(state) {
        if (Math.random() <= this.epsilon) {
            return Math.floor(Math.random() * this.actionSize);
        }
        const tensorState = tf.tensor2d([state], [1, this.stateSize]);
        const predictions = this.model.predict(tensorState);
        const action = (await predictions.data()).indexOf(Math.max(...predictions.dataSync()));
        return action;
    }
}

const agent = new DQNAgent(4, 4);

// Train API Route
app.post("/train", async (req, res) => {
    try {
        const { file_path } = req.body;
        if (!file_path) {
            return res.status(400).json({ error: "File path is required" });
        }

        const dataset = await loadAndPreprocessDataset(file_path);

        // Calculate QoS for each row
        dataset.forEach(row => {
            row.QoSScore = calculateQoS(row.EndToEndDelay, row.PacketDeliveryRate);
        });

        res.json({ qos_scores: dataset.map(row => row.QoSScore) });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
