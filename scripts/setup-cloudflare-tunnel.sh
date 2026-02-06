#!/bin/bash
# HDRit Backend - Cloudflare Tunnel Setup
# This script exposes your Mac Studio backend to the internet

echo "⚡ HDRit Cloudflare Tunnel Setup"
echo "================================="

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "📦 Installing cloudflared..."
    brew install cloudflared
fi

echo ""
echo "🚀 Starting Cloudflare Tunnel for HDRit backend..."
echo "   This will expose http://localhost:8000 to the internet"
echo ""

# Start the tunnel (quick tunnel - no account needed)
# This generates a random subdomain on trycloudflare.com
cloudflared tunnel --url http://localhost:8000

# Note: The tunnel URL will be printed to the console
# Copy it and add to Vercel environment variables as NEXT_PUBLIC_BACKEND_URL
