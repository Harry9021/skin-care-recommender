const express = require('express');
const router = express.Router();
const recommendationController = require('../controllers/recommendationController');

// Route to get recommendations
router.post('/', recommendationController.getRecommendation);

module.exports = router;
