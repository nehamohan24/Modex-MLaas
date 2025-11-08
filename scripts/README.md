# AWS Setup Scripts

This directory contains scripts to help set up and configure AWS services for the MLaaS platform.

## AWS S3 Setup

### Prerequisites

1. **AWS CLI installed and configured:**
   ```bash
   # Install AWS CLI
   pip install awscli
   
   # Configure AWS credentials
   aws configure
   ```

2. **Required AWS permissions:**
   - S3: CreateBucket, PutBucketPolicy, PutBucketVersioning, PutObject, GetObject
   - IAM: GetCallerIdentity (for verification)

### Setup Steps

1. **Run the setup script:**
   ```bash
   cd scripts
   python aws_setup.py
   ```

2. **Set environment variables:**
   ```bash
   export AWS_BUCKET_NAME=mlaas-platform-data
   export AWS_REGION=us-east-1
   ```

3. **Verify setup:**
   - Check that the bucket was created in AWS Console
   - Verify folder structure exists
   - Test upload/download functionality

### Manual Setup (Alternative)

If you prefer to set up AWS S3 manually:

1. **Create S3 Bucket:**
   - Go to AWS S3 Console
   - Create bucket with name: `mlaas-platform-data`
   - Choose region: `us-east-1` (or your preferred region)

2. **Configure Bucket Settings:**
   - Enable versioning
   - Set up encryption (AES-256)
   - Configure access policies

3. **Create Folder Structure:**
   ```
   mlaas-platform-data/
   ├── datasets/
   ├── models/
   ├── logs/
   └── temp/
   ```

4. **Set Environment Variables:**
   ```env
   AWS_ACCESS_KEY_ID=your-access-key
   AWS_SECRET_ACCESS_KEY=your-secret-key
   AWS_BUCKET_NAME=mlaas-platform-data
   AWS_REGION=us-east-1
   ```

### Troubleshooting

**Common Issues:**

1. **"AWS credentials not found"**
   - Run `aws configure` to set up credentials
   - Or set environment variables: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

2. **"Access Denied" errors**
   - Check IAM permissions for S3 access
   - Ensure bucket policy allows your account

3. **"Bucket already exists"**
   - This is normal if you've run the script before
   - The script will continue with other setup steps

4. **"Invalid region" errors**
   - Ensure the region is valid and accessible
   - Some regions may require special permissions

### Security Best Practices

1. **Use IAM Roles** (recommended for production):
   - Create IAM role with minimal required permissions
   - Attach role to EC2 instances or Lambda functions

2. **Enable MFA** for AWS account access

3. **Use Bucket Policies** to restrict access

4. **Enable CloudTrail** for audit logging

5. **Regular Security Audits** of S3 permissions

### Cost Optimization

1. **Lifecycle Policies:**
   - Move old data to cheaper storage classes
   - Delete temporary files automatically

2. **Monitoring:**
   - Set up CloudWatch alarms for costs
   - Monitor storage usage

3. **Compression:**
   - Enable compression for stored data
   - Use appropriate file formats

### Production Deployment

For production deployment:

1. **Use separate AWS accounts** for different environments
2. **Implement proper backup strategies**
3. **Set up monitoring and alerting**
4. **Configure disaster recovery procedures**
5. **Regular security audits and updates**
