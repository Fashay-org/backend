Fashay Deployment Guide
This guide covers the deployment process for Fashay application using Docker and AWS.
Table of Contents

Prerequisites
Local Setup
AWS Setup
Docker Commands
Troubleshooting

Prerequisites

Docker installed locally
AWS CLI configured
AWS ECR repository created
EC2 instance running
Security group with necessary ports open

Local Setup
Docker Build and Push
# Build Docker image
docker build -t fashay-app:latest -f backend/Dockerfile .

# Tag image for ECR
docker tag fashay-app:latest 571600837748.dkr.ecr.us-east-1.amazonaws.com/fashay-app:latest

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 571600837748.dkr.ecr.us-east-1.amazonaws.com

# Push to ECR
docker push 571600837748.dkr.ecr.us-east-1.amazonaws.com/fashay-app:latest
AWS Setup
EC2 Configuration

Security Group Settings:

Port 22 (SSH)
Port 80 (HTTP)
Port 443 (HTTPS)
Port 8000 (Application)



Environment Setup on EC2
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 571600837748.dkr.ecr.us-east-1.amazonaws.com

# Create .env file
cat > .env << 'EOL'
OPENAI_API_KEY2=your_key
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
SUPABASE_SERVICE_ROLE_KEY=your_key
LANGSMITH_API_KEY=your_key
LANGCHAIN_ENDPOINT=your_endpoint
QDRANT_URL=your_url
QDRANT_API_KEY=your_key
PINECONE_API_KEY=your_key
EOL
Docker Commands
Basic Operations
# Pull latest image
docker pull 571600837748.dkr.ecr.us-east-1.amazonaws.com/fashay-app:latest

# Run container
docker run -d -p 8000:8000 \
  --env-file .env \
  --workdir /app/backend/backend \
  571600837748.dkr.ecr.us-east-1.amazonaws.com/fashay-app:latest

# Check running containers
docker ps

# Stop container
docker stop [container_id]

# Remove container
docker rm [container_id]
Container Management
# Remove all stopped containers
docker rm $(docker ps -a -q)

# Remove image
docker rmi 571600837748.dkr.ecr.us-east-1.amazonaws.com/fashay-app:latest

# View logs
docker logs -f [container_id]

# Interactive shell
docker exec -it [container_id] /bin/bash

Troubleshooting
Common Issues

Container fails to start:
docker logs [container_id]

Check file structure:
docker run -it --rm [image_id] /bin/bash
ls -la /app/backend

Port issues:

Verify security group settings
Check port mapping in docker run command
Access Application
http://[EC2-PUBLIC-IP]:8000

Maintenance
Update Application
# Build new version locally
docker build -t fashay-app:latest .

# Push to ECR
docker push 571600837748.dkr.ecr.us-east-1.amazonaws.com/fashay-app:latest

# On EC2: Pull and restart
docker pull 571600837748.dkr.ecr.us-east-1.amazonaws.com/fashay-app:latest
docker stop [old_container_id]
docker run -d -p 8000:8000 --env-file .env [new_image_id]

Notes

Always backup .env file
Monitor EC2 disk space
Keep Docker images updated
Regular security group audits
Check application logs for issues