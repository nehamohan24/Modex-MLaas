#!/usr/bin/env python3
"""
MongoDB Setup Script for MLaaS Platform
This script helps set up MongoDB and create the necessary collections.
"""

import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv

def setup_mongodb():
    """Set up MongoDB database and collections"""
    
    # Load environment variables
    load_dotenv()
    
    # Get MongoDB URI
    mongo_uri = os.environ.get('MONGO_URI', 'mongodb+srv://Modex:majorproject@cluster0.t3xszeh.mongodb.net/mlaas_platform?retryWrites=true&w=majority')
    
    try:
        # Connect to MongoDB
        print("🔌 Connecting to MongoDB...")
        client = MongoClient(mongo_uri)
        
        # Test connection
        client.admin.command('ping')
        print("✅ Successfully connected to MongoDB!")
        
        # Get database
        db = client.mlaas_platform
        
        # Create collections with validation schemas
        print("📊 Setting up collections...")
        
        # Users collection
        try:
            db.create_collection("users")
            print("✅ Created 'users' collection")
        except Exception as e:
            print(f"ℹ️  'users' collection already exists: {e}")
        
        # Datasets collection
        try:
            db.create_collection("datasets")
            print("✅ Created 'datasets' collection")
        except Exception as e:
            print(f"ℹ️  'datasets' collection already exists: {e}")
        
        # Models collection
        try:
            db.create_collection("models")
            print("✅ Created 'models' collection")
        except Exception as e:
            print(f"ℹ️  'models' collection already exists: {e}")
        
        # Create indexes for better performance
        print("🔍 Creating indexes...")
        
        # Users indexes
        db.users.create_index("username", unique=True)
        db.users.create_index("email", unique=True)
        print("✅ Created indexes for 'users' collection")
        
        # Datasets indexes
        db.datasets.create_index("user_id")
        db.datasets.create_index([("domain", 1), ("subdomain", 1)])
        db.datasets.create_index("uploaded_at")
        print("✅ Created indexes for 'datasets' collection")
        
        # Models indexes
        db.models.create_index([("domain", 1), ("subdomain", 1)])
        db.models.create_index("created_by")
        db.models.create_index("created_at")
        print("✅ Created indexes for 'models' collection")
        
        # Get collection stats
        print("\n📈 Database Statistics:")
        print(f"Users: {db.users.count_documents({})}")
        print(f"Datasets: {db.datasets.count_documents({})}")
        print(f"Models: {db.models.count_documents({})}")
        
        print("\n🎉 MongoDB setup completed successfully!")
        print("You can now start your Flask application.")
        
    except Exception as e:
        print(f"❌ Error setting up MongoDB: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure MongoDB is running: brew services start mongodb-community")
        print("2. Check your MONGO_URI in .env file")
        print("3. For local MongoDB: mongodb://localhost:27017/mlaas_platform")
        print("4. For MongoDB Atlas: mongodb+srv://username:password@cluster.mongodb.net/mlaas_platform")
        sys.exit(1)

if __name__ == "__main__":
    setup_mongodb()
