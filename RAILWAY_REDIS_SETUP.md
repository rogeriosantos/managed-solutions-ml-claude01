# Railway Deployment Instructions for Redis

## 1. Add Redis to your Railway project:

```bash
# In Railway dashboard:
# 1. Go to your project
# 2. Click "Add Service"
# 3. Select "Database" -> "Redis"
# 4. Railway will automatically create a Redis instance
```

## 2. Railway will automatically set these environment variables:

- `REDIS_URL` - The complete Redis connection URL
- `REDIS_HOST` - Redis hostname
- `REDIS_PORT` - Redis port (usually 6379)
- `REDIS_PASSWORD` - Redis password

## 3. Your application will automatically use the REDIS_URL environment variable

The code in `api/dependencies.py` is already configured to:

- Read `REDIS_URL` from environment variables
- Fall back to mock Redis client if connection fails
- Handle connection timeouts and retries

## 4. To test Redis connection locally:

```bash
# Install Redis locally (optional for testing)
# Windows: Use Redis for Windows or Docker
docker run -d -p 6379:6379 redis:latest

# Set local Redis URL
set REDIS_URL=redis://localhost:6379/0

# Test your application
uvicorn api.main:app --reload
```

## 5. Production Environment Variables for Railway:

Set these in Railway dashboard under "Variables":

```
DEBUG=False
ENVIRONMENT=production
REDIS_URL=<automatically set by Railway Redis addon>
DB_HOST=<your MySQL host>
DB_PORT=3306
DB_NAME=<your database name>
DB_USER=<your MySQL user>
DB_PASSWORD=<your MySQL password>
MODEL_DIR=models
LOG_LEVEL=INFO
```

## 6. Railway Redis Features:

- Automatic backups
- High availability
- Monitoring and metrics
- Auto-scaling
- SSL/TLS encryption
- Persistent storage

Your application is already configured to work with Railway Redis!
