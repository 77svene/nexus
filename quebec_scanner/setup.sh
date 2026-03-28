#!/bin/bash
# Quebec Business Opportunity Scanner - Setup Script
# Run this once to configure the system.

echo "🍁 QUÉBEC BUSINESS OPPORTUNITY SCANNER - SETUP"
echo "================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install it first."
    exit 1
fi
echo "✅ Python: $(python3 --version)"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install -r requirements.txt 2>&1 | tail -3
echo "✅ Dependencies installed"

# Check Ollama
echo ""
if command -v ollama &> /dev/null; then
    echo "✅ Ollama installed"
    if ollama list 2>/dev/null | grep -q "qwen2.5"; then
        echo "✅ Qwen2.5 model available"
    else
        echo "⚠️  Qwen2.5 not found. Run: ollama pull qwen2.5:9b"
    fi
else
    echo "⚠️  Ollama not found (optional - keyword mode will be used)"
    echo "   Install: curl -fsSL https://ollama.com/install.sh | sh"
    echo "   Then: ollama pull qwen2.5:9b"
fi

# Telegram setup
echo ""
echo "─── TELEGRAM SETUP ───"
echo ""
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "Telegram not configured. To enable notifications:"
    echo ""
    echo "1. Open Telegram, search for @BotFather"
    echo "2. Send /newbot and follow instructions"
    echo "3. Copy the bot token"
    echo "4. Start a chat with your bot"
    echo "5. Visit: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates"
    echo "6. Find your chat_id in the JSON response"
    echo ""
    echo "Then set environment variables:"
    echo "  export TELEGRAM_BOT_TOKEN='your_token_here'"
    echo "  export TELEGRAM_CHAT_ID='your_chat_id_here'"
    echo ""
    echo "Or add to ~/.bashrc for persistence:"
    echo "  echo 'export TELEGRAM_BOT_TOKEN=\"your_token\"' >> ~/.bashrc"
    echo "  echo 'export TELEGRAM_CHAT_ID=\"your_chat_id\"' >> ~/.bashrc"
else
    echo "✅ Telegram configured"
fi

# Create data directory
mkdir -p data logs

# Initialize database
echo ""
echo "Initializing database..."
python3 -c "
from storage.knowledge_base import KnowledgeBase
from config import DB_PATH
kb = KnowledgeBase(DB_PATH)
print('✅ Database initialized:', DB_PATH)
"

echo ""
echo "================================================"
echo "SETUP COMPLETE!"
echo ""
echo "Quick test:   python3 main.py --test"
echo "Single scan:  python3 main.py --scan-once"
echo "24/7 mode:    python3 main.py"
echo "View stats:   python3 main.py --stats"
echo ""
echo "For 24/7 background operation:"
echo "  nohup python3 main.py > logs/output.log 2>&1 &"
echo ""
echo "Or use systemd (recommended):"
echo "  sudo cp quebec-scanner.service /etc/systemd/system/"
echo "  sudo systemctl enable quebec-scanner"
echo "  sudo systemctl start quebec-scanner"
echo "================================================"
