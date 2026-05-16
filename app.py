"""
Slack Slash Command Bot - /policy
Receives Slack slash commands, runs RAG on Employee Handbook, returns answer.
"""

import json
import os
import threading
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request

from pdf_rag import get_qa_chain

load_dotenv()

# ─── Flask App ────────────────────────────────────────────────────

app = Flask(__name__)

SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
PORT = int(os.getenv("PORT", 8000))


def post_slack_message(channel: str, text: str):
    """Send a message to a Slack channel using Bot Token."""
    if not SLACK_BOT_TOKEN:
        return
    try:
        requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={
                "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "channel": channel,
                "text": text,
                "mrkdwn": True,
            },
            timeout=10,
        )
    except Exception as e:
        print(f"❌ Slack post error: {e}")


# ─── Slash Command Handler ────────────────────────────────────────

@app.route("/slack/policy", methods=["POST"])
def handle_policy_command():
    """Handle /policy slash command from Slack."""
    user_query = request.form.get("text", "").strip()
    user_name = request.form.get("user_name", "someone")
    channel_id = request.form.get("channel_id", "")
    response_url = request.form.get("response_url", "")

    if not user_query:
        return jsonify({
            "response_type": "ephemeral",
            "text": "请提供问题，例如：`/policy 年假怎么休？`",
        })

    print(f"💬 /policy from @{user_name}: {user_query}")

    # Acknowledge immediately (Slack requires response within 3s)
    # We'll process and respond asynchronously
    threading.Thread(
        target=process_and_respond,
        args=(user_query, channel_id, response_url),
        daemon=True,
    ).start()

    return jsonify({
        "response_type": "ephemeral",
        "text": f"⏳ 正在查询「{user_query}」… 请稍候",
    })


def process_and_respond(user_query: str, channel_id: str, response_url: str):
    """Process the query and send result back to Slack."""
    try:
        chain = get_qa_chain()
        answer = chain.invoke(user_query)
        print(f"✅ Answer: {answer[:100]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
        answer = f"抱歉，处理您的问题时出现错误：{str(e)}"

    slack_text = f"*📋 关于「{user_query}」*\n\n{answer}"

    # Try response_url first (Slack's 3-second hook), fallback to chat.postMessage
    if response_url:
        try:
            requests.post(
                response_url,
                json={
                    "response_type": "in_channel",
                    "text": slack_text,
                    "mrkdwn": True,
                },
                timeout=5,
            )
            return
        except Exception as e:
            print(f"⚠️ response_url failed, using chat.postMessage: {e}")

    # Fallback: use chat.postMessage
    post_slack_message(channel_id, slack_text)


@app.route("/health", methods=["GET"])
def health():
    deepseek_set = bool(os.getenv("DEEPSEEK_API_KEY", ""))
    bot_token_set = bool(os.getenv("SLACK_BOT_TOKEN", ""))
    return jsonify({
        "status": "ok",
        "deepseek_api_key_set": deepseek_set,
        "slack_bot_token_set": bot_token_set,
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
