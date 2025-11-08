#!/bin/bash

# MLaaS Backend Startup Script
echo "🚀 Starting MLaaS Backend with MongoDB..."

# Activate virtual environment
source venv311/bin/activate

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from example..."
    cp env_example.txt .env
    echo "📝 Please edit .env file with your configuration before running again."
    echo "   Especially update MONGO_URI if using MongoDB Atlas."
    exit 1
fi

# Check if MongoDB is running (for local setup)
if grep -q "mongodb://localhost" .env; then
    echo "🔍 Checking MongoDB connection..."
    if ! python -c "from pymongo import MongoClient; MongoClient('mongodb://localhost:27017').admin.command('ping')" 2>/dev/null; then
        echo "❌ MongoDB is not running. Please start MongoDB first:"
        echo "   macOS: brew services start mongodb-community"
        echo "   Ubuntu: sudo systemctl start mongod"
        echo "   Or use MongoDB Atlas (cloud) and update MONGO_URI in .env"
        exit 1
    fi
    echo "✅ MongoDB connection successful!"
fi

# Run setup script if needed
echo "🔧 Running MongoDB setup..."
python setup_mongodb.py

# Start the Flask application
echo "🌟 Starting Flask application..."
python app.py
