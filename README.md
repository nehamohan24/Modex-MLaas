# MLaaS Platform - Collaborative Machine Learning with Encrypted Data

A comprehensive Machine Learning as a Service (MLaaS) platform that enables multiple users to collaboratively train ML models while maintaining data privacy through encryption and federated learning.

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
