# MongoDB Setup Guide for MLaaS Platform

This guide will help you set up MongoDB for persistent data storage in your MLaaS platform.

## Prerequisites

- Python 3.11+
- MongoDB installed locally or MongoDB Atlas account

## Installation Options

### Option 1: Local MongoDB (Recommended for Development)

#### macOS (using Homebrew)
```bash
# Install MongoDB Community Edition
brew tap mongodb/brew
brew install mongodb-community

# Start MongoDB service
brew services start mongodb-community

# Verify installation
mongosh --version
```

#### Ubuntu/Debian
```bash
# Import MongoDB public key
wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add -

# Create list file
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# Install MongoDB
sudo apt-get update
sudo apt-get install -y mongodb-org

# Start MongoDB
sudo systemctl start mongod
sudo systemctl enable mongod
```

#### Windows
1. Download MongoDB Community Server from [mongodb.com](https://www.mongodb.com/try/download/community)
2. Run the installer
3. Start MongoDB service

### Option 2: MongoDB Atlas (Cloud)

1. Go to [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a free account
3. Create a new cluster
4. Get your connection string
5. Update the `MONGO_URI` in your `.env` file

## Setup Steps

### 1. Install Python Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
# Copy the example environment file
cp env_example.txt .env

# Edit .env file with your configuration
nano .env
```

Update the following variables in `.env`:
```env
# For local MongoDB
MONGO_URI=mongodb://localhost:27017/mlaas_platform

# For MongoDB Atlas
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/mlaas_platform?retryWrites=true&w=majority

# Change these for security
SECRET_KEY=your-super-secret-key-change-this
JWT_SECRET_KEY=your-jwt-secret-key-change-this
```

### 3. Run Setup Script
```bash
python setup_mongodb.py
```

This script will:
- Connect to MongoDB
- Create necessary collections (users, datasets, models)
- Set up indexes for better performance
- Display database statistics

### 4. Start Your Application
```bash
python app.py
```

## Database Schema

### Users Collection
```javascript
{
  _id: ObjectId,
  username: String (unique),
  email: String (unique),
  password: String (hashed),
  domain: String,
  subdomain: String,
  created_at: Date,
  datasets: [String], // Array of dataset IDs
  model_contributions: [String] // Array of model IDs
}
```

### Datasets Collection
```javascript
{
  _id: ObjectId,
  user_id: String,
  domain: String,
  subdomain: String,
  filename: String,
  file_content: Binary,
  size: Number,
  file_type: String,
  uploaded_at: Date
}
```

### Models Collection
```javascript
{
  _id: ObjectId,
  domain: String,
  subdomain: String,
  model_type: String,
  accuracy: Number,
  r2_score: Number,
  mse: Number,
  training_data_count: Number,
  contributors: [String], // Array of user IDs
  contributors_count: Number,
  created_by: String,
  created_at: Date
}
```

## API Endpoints

### New Endpoints Added:
- `GET /api/stats` - Get global and user statistics
- `GET /api/models/<domain>/<subdomain>` - Get models by domain/subdomain

### Updated Endpoints:
- All endpoints now use MongoDB for persistent storage
- Data persists between server restarts
- Global statistics track all users' contributions

## Troubleshooting

### Connection Issues
```bash
# Check if MongoDB is running
brew services list | grep mongodb

# Start MongoDB if not running
brew services start mongodb-community

# Test connection
mongosh
```

### Permission Issues
```bash
# Make sure MongoDB has proper permissions
sudo chown -R mongodb:mongodb /var/lib/mongodb
sudo chown -R mongodb:mongodb /var/log/mongodb
```

### Port Issues
```bash
# Check if port 27017 is in use
lsof -i :27017

# Kill process if needed
sudo kill -9 <PID>
```

## Benefits of MongoDB Integration

1. **Persistent Storage**: Data survives server restarts
2. **Scalability**: Can handle multiple users and large datasets
3. **Global Statistics**: Track total datasets, models, and users
4. **Data Relationships**: Proper linking between users, datasets, and models
5. **Indexing**: Fast queries with proper database indexes
6. **Flexibility**: Easy to add new fields and features

## Next Steps

1. Test the application with multiple users
2. Upload datasets and train models
3. Check the global statistics endpoint
4. Consider adding data validation and encryption
5. Set up regular database backups
