#!/bin/bash
# HDRit Production Startup Script
# Starts backend + ngrok tunnel with custom domain

echo "⚡ Starting HDRit Production Server..."

# Kill any existing processes
pkill -f "uvicorn src.api.main:app" 2>/dev/null
pkill -f ngrok 2>/dev/null
sleep 2

# Start backend
echo "🚀 Starting FastAPI backend on port 8000..."
cd /private/tmp/autohdr-clone
nohup python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
sleep 3

# Verify backend
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend running"
else
    echo "❌ Backend failed to start"
    exit 1
fi

# Start ngrok with custom domain
echo "🌐 Starting ngrok tunnel..."
nohup ngrok http 8000 --domain=hdr.it.com.ngrok.pro > /tmp/ngrok.log 2>&1 &
sleep 3

# Verify tunnel
if curl -s https://hdr.it.com.ngrok.pro/health > /dev/null; then
    echo "✅ Tunnel active: https://hdr.it.com.ngrok.pro"
else
    echo "⚠️  Tunnel may still be starting..."
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  HDRit Production Server Running                         ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Local:    http://localhost:8000                         ║"
echo "║  LAN:      http://192.168.1.147:8000                     ║"
echo "║  Public:   https://hdr.it.com.ngrok.pro                  ║"
echo "║  Frontend: https://hdr.it.com                            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Logs: tail -f /tmp/backend.log /tmp/ngrok.log"
