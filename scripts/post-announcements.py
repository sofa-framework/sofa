#!python
import os
import requests

# Recover all info as env var
discord_token = os.environ['DISCORD_ANNOUNCEMENT_WEBHOOK_URL']
discussion_name = os.environ['DISCUSSION_NAME']
discussion_number = os.environ['DISCUSSION_NUMBER']
discussion_author = os.environ['DISCUSSION_AUTHOR']

# Create the message
message = "New post: ["+str(discussion_name)+"](https://github.com/sofa-framework/sofa/discussions/"+str(discussion_number)+")\nby "+str(discussion_author)

# Send message to Discord room
payload = {'content': message}
response = requests.post(discord_token, json=payload)
print(response)