#!/bin/bash

echo "🚀 Starting MLaaS Platform Servers..."

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $1 is already in use"
        return 1
    else
        return 0
    fi
}

# Start Backend
echo "📡 Starting Backend Server..."
cd backend
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

source venv/bin/activate
if check_port 5000; then
    python app.py &
    BACKEND_PID=$!
    echo "✅ Backend started on http://localhost:5000 (PID: $BACKEND_PID)"
else
    echo "⚠️  Backend port 5000 is busy, trying to use existing process"
fi

# Start Frontend
echo "🌐 Starting Frontend Server..."
cd ../frontend

if check_port 3000; then
    PORT=3000 BROWSER=none npm start &
    FRONTEND_PID=$!
    echo "✅ Frontend started on http://localhost:3000 (PID: $FRONTEND_PID)"
elif check_port 3001; then
    PORT=3001 BROWSER=none npm start &
    FRONTEND_PID=$!
    echo "✅ Frontend started on http://localhost:3001 (PID: $FRONTEND_PID)"
else
    echo "⚠️  Both ports 3000 and 3001 are busy, trying to use existing process"
fi

echo ""
echo "🎉 MLaaS Platform is running!"
echo "📱 Frontend: http://localhost:3000 or http://localhost:3001"
echo "🔧 Backend API: http://localhost:5000"
echo "🧪 Test Page: http://localhost:8080/test.html"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user interrupt
trap 'echo "🛑 Stopping servers..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT
wait
