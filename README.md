# Modex: A Machine Learning as a Service (MLaaS) platform

A platform designed to make machine learning accessible, secure, and collaborative for users who may not have deep ML expertise.
At a high level, Modex allows users to upload datasets, train machine-learning models, and run predictions through simple interfaces, without writing ML code.
##Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- MongoDB
- AWS Account (for S3 storage)

### Backend Setup
```bash
cd backend
pip install -r requirements.txt

# Set up environment variables
cp env_example.txt .env
# Edit .env with your configuration

# Start MongoDB (if not running)
mongod

# Run the Flask application
python app.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```
