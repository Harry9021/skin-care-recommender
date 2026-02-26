# 🧴 Skincare Recommendation System

> **AI-Powered Personalized Skincare Product Recommendations | Professional & Production-Ready**

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![React](https://img.shields.io/badge/react-18+-61dafb)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Overview

A **full-stack machine learning application** that provides personalized skincare product recommendations based on skin type and up to three skin concerns. Features a professional Python/Flask backend with ML ensemble model and responsive React frontend.

### ✨ Key Features

✅ **AI-Powered Recommendations** - Ensemble model (KNN + Random Forest) with 82% accuracy
✅ **Production-Ready Backend** - Modular architecture with separation of concerns  
✅ **Responsive UI** - Mobile-first design (320px to 1920px)
✅ **Google OAuth 2.0** - Ready for social login integration
✅ **REST API** - Professional endpoints with error handling
✅ **Input Validation** - Comprehensive validation at all layers
✅ **Structured Logging** - Professional logging system
✅ **Environment Config** - Safe secrets management with .env

---

## 🛠 Tech Stack

### Frontend
- React 18 with Create React App
- Tailwind CSS (utility-first styling)
- React Router (SPA routing)
- @react-oauth/google (OAuth integration)

### Backend
- Flask 2.3.3 (Web framework)
- scikit-learn 1.3.0 (ML algorithms)
- pandas 2.0.3 (Data processing)
- numpy 1.24.3 (Numerical computing)
- imbalanced-learn 0.11.0 (SMOTE)
- PyJWT 2.8.1 (JWT authentication)
- python-dotenv 1.0.0 (Environment management)

---

## 📂 Project Structure

```
skin-care-recommender/
│
├── ml_model/                          # Backend - ML & API
│   ├── config/
│   │   └── settings.py               # Configuration management
│   ├── models/
│   │   └── recommendation.py         # ML model logic
│   ├── routes/
│   │   ├── health.py                 # Health check endpoints
│   │   ├── recommendations.py        # Recommendation API
│   │   └── auth.py                   # OAuth endpoints
│   ├── utils/
│   │   ├── logger.py                 # Logging setup
│   │   ├── errors.py                 # Custom exceptions
│   │   └── validators.py             # Input validation
│   ├── middleware/
│   │   └── auth_middleware.py        # Auth decorators
│   ├── app.py                        # Main Flask app
│   ├── requirements.txt               # Dependencies
│   ├── .env.example                  # Config template
│   └── to_be_use_dataset.csv         # Training dataset
│
├── ui/                               # Frontend - React
│   ├── src/
│   │   ├── components/
│   │   │   ├── Formpage.jsx          # Input form
│   │   │   ├── Results.jsx           # Recommendations
│   │   │   ├── CartPage.jsx          # Shopping cart
│   │   │   ├── Home.jsx              # Home page
│   │   │   ├── Profile.jsx           # User profile
│   │   │   ├── cards/
│   │   │   │   └── Resultcard.jsx    # Product card
│   │   │   ├── context/
│   │   │   │   └── CartContext.jsx   # State management
│   │   │   └── Router/
│   │   │       └── Router.jsx        # Routes
│   │   ├── Styles/                   # Component styles
│   │   └── Vectors/                  # Images & icons
│   ├── package.json
│   └── tailwind.config.js
│
└── README.md                         # This file
```

---

## 🚀 Quick Start

### Fastest Way to Start
```bash
npm start
```

This single command will:
- Check Node.js and Python installation
- Create Python virtual environment
- Install dependencies (npm + pip)
- Create env files from templates
- Launch backend and frontend together

After startup:
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:5000`

### Prerequisites
- Python 3.8+
- Node.js 14+
- Git

### Alternative Setup Methods

#### Development Mode (already setup)
```bash
npm run dev
```

#### Backend only
```bash
npm run setup:backend
npm run model
```

#### Frontend only
```bash
npm run setup:frontend
npm run client
```

#### Manual setup
```bash
# Backend
cd ml_model
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
python app.py

# Frontend (separate terminal)
cd ui
npm install
npm start
```

### Available Commands
```bash
npm start               # Setup everything and launch
npm run setup           # Same as npm start
npm run dev             # Launch if already setup
npm run setup:backend   # Setup backend only
npm run setup:frontend  # Setup frontend only
npm run model           # Run backend without setup
npm run client          # Run frontend without setup
npm run build           # Build frontend for production
```

---

## 📡 API Endpoints

### POST `/api/recommend`
Get personalized product recommendations.

**Request:**
```json
{
  "skin_type": "oily",
  "concern_1": "acne",
  "concern_2": "excess-oil",
  "concern_3": "sensitivity",
  "top_n": 10
}
```

**Response (200):**
```json
{
  "status": "success",
  "count": 10,
  "recommendations": [
    {
      "rank": 1,
      "label": "sunscreen",
      "brand": "Neutrogena",
      "name": "Ultra Sheer Sunscreen SPF 50+",
      "price": 7.99,
      "confidence": 0.87
    }
  ],
  "timestamp": "2024-02-26T10:30:00.000Z"
}
```

### GET `/api/categories`
Get all available skin types and concerns.

**Response:**
```json
{
  "status": "success",
  "categories": {
    "skin type": ["oily", "dry", "normal", ...],
    "concern": ["acne", "sensitivity", ...],
    "concern 2": [...],
    "concern 3": [...]
  },
  "summary": {
    "total_skin_types": 6,
    "total_products": 225
  }
}
```

### GET `/api/model-info`
Get model performance metrics.

**Response:**
```json
{
  "status": "success",
  "model_info": {
    "model_type": "Ensemble (KNN + Random Forest)",
    "is_trained": true,
    "metrics": {
      "accuracy": 0.82,
      "precision": 0.81,
      "recall": 0.82,
      "f1": 0.81
    },
    "feature_importance": {
      "skin type": 0.35,
      "concern": 0.28,
      "concern 2": 0.20,
      "concern 3": 0.17
    }
  }
}
```

### GET `/health`
Health check endpoint.

---

## 🔐 Google OAuth Setup

### Get Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create new project
3. Enable "Google+ API"
4. Credentials → Create OAuth 2.0 Client ID
5. Web application
6. Add authorized redirect URIs:
   - `http://localhost:3000`
   - `http://localhost:5000`
7. Copy Client ID and Secret

### Configure

Edit `ml_model/.env`:
```env
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret
JWT_SECRET=your-super-secret-key
```

### OAuth Endpoints

- **POST** `/api/auth/google` - Exchange Google token for JWT
- **POST** `/api/auth/verify-token` - Verify JWT
- **POST** `/api/auth/refresh-token` - Refresh token

---

## 🤖 ML Model

### Algorithm Details

| Aspect | Details |
|--------|---------|
| **Type** | Ensemble (KNN + Random Forest) |
| **Training Data** | 225 skincare products |
| **Features** | Skin type + 3 concerns |
| **Accuracy** | 82% |
| **Validation** | Stratified K-Fold (3 splits) |
| **Class Balancing** | SMOTE |

### Dataset

6 product categories:
- Sunscreen (SPF Products)
- Foundation (Base Makeup)
- Cleanser (Face Cleansing)
- Face Moisturizers (Hydration)
- Concealer (Coverage)
- Mask & Peel (Treatments)

### Retraining

```python
from models.recommendation import RecommendationModel
from config.settings import active_config

model = RecommendationModel()
model.load_dataset(active_config.DATASET_PATH)
metrics = model.train()
model.save(active_config.MODEL_PATH)

print(f"Accuracy: {metrics['accuracy']:.4f}")
```

---

## ⚙️ Backend Architecture

### Professional Design

**Modular Structure:**
- `config/` - Centralized configuration
- `models/` - ML business logic
- `routes/` - API endpoints
- `utils/` - Helper functions
- `middleware/` - Cross-cutting concerns

**Error Handling:**
- Custom exceptions with HTTP codes
- Consistent error format
- Input validation layers

**Logging:**
- Structured logging
- Multiple severity levels
- File-based persistence

**Security:**
- Environment-based secrets
- JWT authentication
- Input sanitization

### Request Flow

```
Client Request → Flask Route → Input Validation → 
ML Model → Error Handling → Response → Logging
```

---

## 📱 Frontend Features

### Responsive Design

| Device | Layout |
|--------|--------|
| **Mobile** (320px+) | Single column |
| **Tablet** (768px+) | Two column |
| **Desktop** (1024px+) | Full featured |

### Components

- **Formpage** - Skin type & concerns input
- **Results** - Recommendation display
- **ResultCard** - Product details
- **CartPage** - Shopping cart
- **Profile** - User preferences

### Styling

- Tailwind CSS utilities
- CSS Grid layouts
- CSS animations
- Mobile-first approach

---

## ⚙️ Environment Configuration

Create `ml_model/.env`:

```env
# Flask
FLASK_ENV=development
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# CORS
CORS_ORIGINS=http://localhost:3000

# Model
MODEL_PATH=skincare_model_enhanced.pkl
DATASET_PATH=to_be_use_dataset.csv
DEFAULT_TOP_N=10

# Logging
LOG_LEVEL=DEBUG
LOG_FILE=logs/app.log

# Google OAuth
GOOGLE_CLIENT_ID=your_id
GOOGLE_CLIENT_SECRET=your_secret

# JWT
JWT_SECRET=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION=86400
```

---

## ❌ Error Handling

### Response Format

```json
{
  "status": "error",
  "error_code": "ERROR_CODE",
  "message": "Human readable message",
  "details": {
    "field": "additional context"
  }
}
```

### Common Codes

| Code | HTTP | Description |
|------|------|-------------|
| `INVALID_INPUT` | 400 | Missing/invalid fields |
| `INVALID_CATEGORY` | 400 | Invalid skin type/concern |
| `MODEL_NOT_READY` | 503 | Model initializing |
| `AUTH_FAILED` | 401 | Authentication failed |
| `FORBIDDEN` | 403 | Not authorized |

---

## 🚀 Deployment

### Production with Gunicorn

```bash
cd ml_model
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker

```bash
docker build -t skincare-api .
docker run -p 5000:5000 skincare-api
```

### Production Checklist

- [ ] Set `FLASK_ENV=production`
- [ ] Use strong `JWT_SECRET`
- [ ] Enable HTTPS/SSL
- [ ] Proper CORS origins
- [ ] File-based logging
- [ ] Gunicorn/uWSGI server
- [ ] Monitoring & alerts
- [ ] Database for users (future)

---

## 📊 Performance

### Backend
- Response time: 200-500ms
- Throughput: 100+ req/sec
- Model load: 1-2 seconds

### Frontend
- Lighthouse: 85+
- Load time: <2 seconds
- FCP: <1 second

---

## 🐛 Troubleshooting

### Backend

**Port in use:**
```bash
# Change in .env
FLASK_PORT=5001
python app.py
```

**CORS errors:**
```bash
# Update CORS_ORIGINS in .env
CORS_ORIGINS=http://localhost:3000
```

**Model not loading:**
```bash
# Check dataset exists
ls ml_model/to_be_use_dataset.csv
# Check logs
tail -f ml_model/logs/app.log
```

### Frontend

**API connection failed:**
```bash
# Verify backend
curl http://localhost:5000/health
```

**Port 3000 in use:**
```bash
PORT=3001 npm start
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'feat: add feature'`
4. Push to fork: `git push origin feature/amazing-feature`
5. Open Pull Request

### Code Standards

- Follow PEP 8 (Python)
- Use functional components (React)
- Add docstrings/comments
- Meaningful commit messages

---

## 📄 License

MIT License - Free for personal and commercial use.

---

## 🙏 Acknowledgments

- scikit-learn - ML algorithms
- Flask - Web framework
- React - UI library
- Tailwind CSS - Styling

---

## 📞 Support

- 📖 Check README documentation
- 🐛 Report issues on [GitHub Issues](https://github.com/Harry9021/skin-care-recommender/issues)
- 💬 Start [discussion](https://github.com/Harry9021/skin-care-recommender/discussions)

**Contact**: [@Harry9021](https://github.com/Harry9021)

---

## 🎓 What You Learn

✅ Full-Stack Development (React + Flask)
✅ Machine Learning (Ensemble models)
✅ Professional Architecture
✅ REST API Design
✅ OAuth 2.0 Integration
✅ DevOps (Docker, Environment management)
✅ Responsive UI/UX

---

**Made with ❤️ by [@Harry9021](https://github.com/Harry9021)**

⭐ If helpful, please star the repository!

---

*Version: 2.0.0 (Professional Edition)*
*Last Updated: February 26, 2024*
