"""
Slack Slash Command Bot - /policy
Receives Slack slash commands, runs RAG on Employee Handbook, returns answer.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from pdf_rag import get_qa_chain

load_dotenv()

# ─── Flask App ────────────────────────────────────────────────────

app = Flask(__name__)

SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
PORT = int(os.getenv("PORT", 8000))

# ─── Slash Command Handler ────────────────────────────────────────

@app.route("/slack/policy", methods=["POST"])
def handle_policy_command():
    """Handle /policy slash command from Slack."""
    user_query = request.form.get("text", "").strip()
    user_name = request.form.get("user_name", "someone")

    if not user_query:
        return jsonify({
            "response_type": "ephemeral",
            "text": "请提供问题，例如：`/policy 年假怎么休？`",
        })

    print(f"💬 /policy from @{user_name}: {user_query}")

    try:
        chain = get_qa_chain()
        answer = chain.invoke(user_query)
        print(f"✅ Answer: {answer[:100]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
        answer = f"抱歉，处理您的问题时出现错误：{str(e)}"

    return jsonify({
        "response_type": "in_channel",
        "text": f"*📋 关于「{user_query}」*\n\n{answer}",
        "mrkdwn": True,
    })


@app.route("/health", methods=["GET"])
def health():
    import os
    deepseek_set = bool(os.getenv("DEEPSEEK_API_KEY", ""))
    return jsonify({
        "status": "ok",
        "deepseek_api_key_set": deepseek_set,
    })


# ─── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pre-warm the QA chain on startup
    print("🔄 Pre-warming QA engine...")
    try:
        get_qa_chain()
    except Exception as e:
        print(f"⚠️  QA engine init warning: {e}")
    print(f"🚀 Server starting on http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)

