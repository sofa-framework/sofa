#!python
import os
import requests


# Recover all info as env var
discord_token = os.environ['DISCORD_WEBHOOK_URL']
message = os.environ['MESSAGE']
bot_name = os.environ['BOT_NAME']
embeds_title = os.environ['EMBEDS_TITLE']
embeds_url = os.environ['EMBEDS_URL']
embeds_description = os.environ['EMBEDS_DESCRIPTION']


# Format message
data = {
    "content" : message,
    "username" : bot_name
}

if embeds_title == "":
    data["flags"] = 4
elif:
    data["embeds"] = [
        {
            "description" : embeds_description,
            "title" : embeds_title,
            "type" : "rich",
            "url" : embeds_url,
            "color" : "15224347"
        }
    ]

# Send message to Discord
response = requests.post(discord_token, json=data)
print("Status: "+str(response.status_code)+"\nReason: "+str(response.reason)+"\nText: "+str(response.text))

