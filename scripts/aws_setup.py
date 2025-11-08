#!/usr/bin/env python3
"""
AWS S3 Setup Script for MLaaS Platform
This script helps set up the AWS S3 bucket and configure permissions
"""

import boto3
import json
import os
from botocore.exceptions import ClientError, NoCredentialsError

def create_s3_bucket(bucket_name, region='us-east-1'):
    """Create S3 bucket with proper configuration"""
    try:
        s3_client = boto3.client('s3', region_name=region)
        
        # Check if bucket already exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✅ Bucket '{bucket_name}' already exists")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] != '404':
                print(f"❌ Error checking bucket: {e}")
                return False
        
        # Create bucket
        if region == 'us-east-1':
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        
        print(f"Bucket '{bucket_name}' created successfully in region '{region}'")
        return True
        
    except NoCredentialsError:
        print("AWS credentials not found. Please configure AWS CLI or set environment variables.")
        return False
    except ClientError as e:
        print(f"Error creating bucket: {e}")
        return False

def configure_bucket_policy(bucket_name):
    """Configure bucket policy for security"""
    try:
        s3_client = boto3.client('s3')
        
        # Basic security policy
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "DenyInsecureConnections",
                    "Effect": "Deny",
                    "Principal": "*",
                    "Action": "s3:*",
                    "Resource": [
                        f"arn:aws:s3:::{bucket_name}",
                        f"arn:aws:s3:::{bucket_name}/*"
                    ],
                    "Condition": {
                        "Bool": {
                            "aws:SecureTransport": "false"
                        }
                    }
                }
            ]
        }
        
        s3_client.put_bucket_policy(
            Bucket=bucket_name,
            Policy=json.dumps(policy)
        )
        
        print(f"✅ Security policy applied to bucket '{bucket_name}'")
        return True
        
    except ClientError as e:
        print(f"❌ Error applying bucket policy: {e}")
        return False

def enable_versioning(bucket_name):
    """Enable versioning for the bucket"""
    try:
        s3_client = boto3.client('s3')
        
        s3_client.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={'Status': 'Enabled'}
        )
        
        print(f"✅ Versioning enabled for bucket '{bucket_name}'")
        return True
        
    except ClientError as e:
        print(f"❌ Error enabling versioning: {e}")
        return False

def create_folder_structure(bucket_name):
    """Create initial folder structure"""
    try:
        s3_client = boto3.client('s3')
        
        folders = [
            'datasets/',
            'models/',
            'logs/',
            'temp/'
        ]
        
        for folder in folders:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=folder,
                Body=''
            )
            print(f"✅ Created folder: {folder}")
        
        return True
        
    except ClientError as e:
        print(f"❌ Error creating folder structure: {e}")
        return False

def test_upload(bucket_name):
    """Test upload functionality"""
    try:
        s3_client = boto3.client('s3')
        
        test_data = {
            'test': True,
            'message': 'MLaaS Platform AWS S3 Test',
            'timestamp': '2024-01-01T00:00:00Z'
        }
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key='test/connection_test.json',
            Body=json.dumps(test_data),
            ContentType='application/json',
            ServerSideEncryption='AES256'
        )
        
        print(f"✅ Test upload successful to bucket '{bucket_name}'")
        return True
        
    except ClientError as e:
        print(f"❌ Error testing upload: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 MLaaS Platform AWS S3 Setup")
    print("=" * 40)
    
    # Get configuration
    bucket_name = os.environ.get('AWS_BUCKET_NAME', 'mlaas-platform-data')
    region = os.environ.get('AWS_REGION', 'us-east-1')
    
    print(f"Bucket Name: {bucket_name}")
    print(f"Region: {region}")
    print()
    
    # Check AWS credentials
    try:
        sts_client = boto3.client('sts')
        identity = sts_client.get_caller_identity()
        print(f"✅ AWS credentials valid for account: {identity['Account']}")
        print()
    except NoCredentialsError:
        print("❌ AWS credentials not found. Please run 'aws configure' or set environment variables.")
        return False
    
    # Setup steps
    steps = [
        ("Creating S3 bucket", lambda: create_s3_bucket(bucket_name, region)),
        ("Configuring security policy", lambda: configure_bucket_policy(bucket_name)),
        ("Enabling versioning", lambda: enable_versioning(bucket_name)),
        ("Creating folder structure", lambda: create_folder_structure(bucket_name)),
        ("Testing upload", lambda: test_upload(bucket_name))
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        print(f"📋 {step_name}...")
        if step_func():
            success_count += 1
        print()
    
    # Summary
    print("=" * 40)
    print(f"✅ Setup completed: {success_count}/{len(steps)} steps successful")
    
    if success_count == len(steps):
        print("🎉 AWS S3 setup completed successfully!")
        print()
        print("Next steps:")
        print("1. Update your .env file with AWS credentials")
        print("2. Start the MLaaS platform")
        print("3. Test the AWS integration")
        return True
    else:
        print("⚠️ Some steps failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    main()
