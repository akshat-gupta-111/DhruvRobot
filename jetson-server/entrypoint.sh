#!/bin/bash

echo "🚀 Starting Dhruv Onboard Server..."
# Start the API server in the background
python3 api_server.py &
API_PID=$!

echo "⏳ Waiting for API Server to boot..."
sleep 5

echo "👁️ Starting Dhruv Vision Client..."
# Start the autonomous client in the foreground
python3 client/autonomous_client.py

# If the client crashes or stops, wait for the API server
wait $API_PID
