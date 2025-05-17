const express = require('express');
const cors = require('cors');
const { exec } = require('child_process');
const app = express();
app.use(cors());
app.use(express.json());

app.post('/predict', (req, res) => {
    const inputData = JSON.stringify(req.body);
    console.log('Received input data:', inputData);  // Log input data
    // Call Python script here
    const escapedInputData = inputData.replace(/"/g, '\\"');
    const command = `python soil_fertility_model.py "${escapedInputData}"`;
    console.log('Executing command:', command);
    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error: ${error.message}`);
            return res.status(500).send('Error processing request');
        }
        console.log('Prediction result:', stdout.trim());  // Log the prediction result
        try {
            const parsed = JSON.parse(stdout.trim());
            res.json(parsed);
        } catch (err) {
            console.error('Failed to parse prediction result:', err);
            res.status(500).send('Invalid prediction output format');
        }
            });
});

app.listen(3001, () => {
    console.log('Server running on port 3001');
});
