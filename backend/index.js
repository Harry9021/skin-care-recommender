// Import necessary modules
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const axios = require('axios'); // Use axios to send requests to Python script

const app = express();
const port = 4000;

// Middleware setup
app.use(bodyParser.json());
app.use(cors());

// Route to receive user input and forward it to the Python script
app.post('/get-recommendations', async (req, res) => {
    const { skin_type, concern_1, concern_2, concern_3 } = req.body; // Get user input

    try {
        // Send a request to the Python script
        const response = await axios.post('http://localhost:5000/recommend', { // URL of your Python service
            skin_type,
            concern_1,
            concern_2,
            concern_3
        });
        console.log(response);

        // Return the response from Python script back to the client
        res.status(200).json({
            message: 'Recommendations fetched successfully!',
            data: response.data  // Recommendations from Python script

        });
    } catch (error) {
        console.error('Error fetching recommendations:', error);
        res.status(500).json({ message: 'Error fetching recommendations' });
    }
});

// Start the server
app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
