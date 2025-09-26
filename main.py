import os
import hmac
import hashlib
import time
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
load_dotenv()
# --- Configuration (from environment variables) ---
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN") # Needed for getting message permalink
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO") # e.g., "your-username/your-repo"



app = Flask(__name__)
def get_user_info(user_id):
    """Gets a user's real name from their Slack ID."""
    url = "https://slack.com/api/users.info"
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    params = {"user": user_id}
    response = requests.get(url, headers=headers, params=params)
    if response.ok:
        data = response.json()
        if data.get("ok"):
            # Return the user's real name, or their display name as a fallback
            return data["user"]["profile"].get("real_name", data["user"]["profile"].get("display_name"))
    return None

def get_channel_info(channel_id):
    """Gets a channel's name from its ID."""
    url = "https://slack.com/api/conversations.info"
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    params = {"channel": channel_id}
    response = requests.get(url, headers=headers, params=params)
    if response.ok:
        data = response.json()
        if data.get("ok"):
            return data["channel"].get("name")
    return None


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
    """
    text = event.get('text', '')
    if not text.upper().startswith("ISSUE:"):
        return None

    title = text[6:].strip()
    user_id = event.get('user')
    channel_id = event.get('channel')
    message_ts = event.get('ts')

    # --- NEW: Fetch descriptive names ---
    user_name = get_user_info(user_id) or user_id
    channel_name = get_channel_info(channel_id) or channel_id
    permalink = get_message_permalink(channel_id, message_ts)

    # --- UPDATED: New body_template with real names ---
    issue_body = (
        f"**Description:**\n{title}\n\n"
        f"--- \n"
        f"### Slack Context\n"
        f"- **Reporter:** {user_name} (`{user_id}`)\n"
        f"- **Channel:** #{channel_name} (`{channel_id}`)\n"
        f"- **[View Original Conversation]({permalink})**"
    )

    mcp_object = {
        "source_system": "slack",
        "context": {
            "event_type": "channel_message",
            "user_id": user_id,
            "user_name": user_name, # Storing the real name
            "channel_id": channel_id,
            "channel_name": channel_name, # Storing the real name
            "timestamp": message_ts,
            "permalink": permalink
        },
        "model": {
            "title": title,
            "body_template": issue_body, # Using the new descriptive body
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