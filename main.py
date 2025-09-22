import os
import hmac
import hashlib
import time
from flask import Flask, request, jsonify
import requests

# --- Configuration (from environment variables) ---
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN") # Needed for getting message permalink
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO") # e.g., "your-username/your-repo"

app = Flask(__name__)

# Note: The verify_slack_request function remains the same as in the previous answer.
# ... (include the verify_slack_request function here) ...
def verify_slack_request(request):
    """Verifies the request signature from Slack."""
    timestamp = request.headers.get('X-Slack-Request-Timestamp')
    slack_signature = request.headers.get('X-Slack-Signature')
    if not timestamp or not slack_signature or abs(time.time() - int(timestamp)) > 60 * 5:
        return False
    req_body = request.get_data(as_text=True)
    base_string = f"v0:{timestamp}:{req_body}".encode('utf-8')
    my_signature = 'v0=' + hmac.new(SLACK_SIGNING_SECRET.encode('utf-8'), base_string, hashlib.sha256).hexdigest()
    return hmac.compare_digest(my_signature, slack_signature)


def get_message_permalink(channel_id, message_ts):
    """Gets a permalink for a Slack message."""
    url = "https://slack.com/api/chat.getPermalink"
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    params = {"channel": channel_id, "message_ts": message_ts}
    response = requests.get(url, headers=headers, params=params)
    if response.ok and response.json().get("ok"):
        return response.json().get("permalink")
    return None

def build_mcp_from_slack_event(event):
    """
    Translates a raw Slack event into our internal Model-Context Protocol object.
    This is the heart of the "Protocol" implementation.
    """
    text = event.get('text', '')
    if not text.upper().startswith("ISSUE:"):
        return None

    title = text[6:].strip()
    channel_id = event.get('channel')
    message_ts = event.get('ts')
    
    permalink = get_message_permalink(channel_id, message_ts)

    mcp_object = {
        "source_system": "slack",
        "context": {
            "event_type": "channel_message",
            "user_id": event.get('user'),
            "channel_id": channel_id,
            "timestamp": message_ts,
            "permalink": permalink
        },
        "model": {
            "title": title,
            "body_template": (
                f"Issue reported from Slack.\n\n"
                f"**Details:**\n> {title}\n\n"
                f"**Context:**\n"
                f"- **Reporter:** <@{event.get('user')}>\n"
                f"- **Channel:** <#{channel_id}>\n"
                f"- **[Link to Conversation]({permalink})**"
            ),
            "suggested_labels": ["bug", "from-slack"]
        }
    }
    return mcp_object

def create_github_issue_from_mcp(mcp):
    """
    Creates a GitHub issue using a structured MCP object.
    This function only knows about the MCP format, not about Slack.
    """
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    
    # Use the model data from the MCP to build the GitHub payload
    payload = {
        "title": mcp["model"]["title"],
        "body": mcp["model"]["body_template"], # The body is already formatted!
        "labels": mcp["model"]["suggested_labels"]
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 201:
        print(f"Successfully created GitHub issue from MCP: {mcp['model']['title']}")
        return response.json()
    else:
        print(f"Failed to create GitHub issue: {response.status_code} {response.text}")
        return None


@app.route('/slack/events', methods=['POST'])
def slack_events():
    if request.json and request.json.get("type") == "url_verification":
        return jsonify({"challenge": request.json.get("challenge")})
    
    if not verify_slack_request(request):
        return "Invalid signature", 403

    event = request.json.get('event', {})
    if event.get('bot_id'):
        return "OK", 200

    # 1. Translate the incoming request into our internal protocol
    mcp_issue_object = build_mcp_from_slack_event(event)
    
    # 2. If the translation was successful, process it
    if mcp_issue_object:
        create_github_issue_from_mcp(mcp_issue_object)

    return "OK", 200

if __name__ == '__main__':
    app.run(port=3000)