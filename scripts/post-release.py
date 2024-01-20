#!python
import os
import requests

# Recover all info as env var
discord_token = os.environ['DISCORD_ANNOUNCEMENTS_WEBHOOK_URL']
release_name = os.environ['RELEASE_NAME']
release_tag = os.environ['RELEASE_TAG']

# Create the message
message = ":loudspeaker: **New SOFA release** :loudspeaker: \n > SOFA "+str(release_name)+": https://github.com/sofa-framework/sofa/releases/tag/"+str(release_tag)+"\n > For more, see the [ChangeLog](https://github.com/sofa-framework/sofa/blob/"+str(release_tag)+"/CHANGELOG.md) \n \n Thanks to all contributors!\n"

# Send message to Discord room
payload = {'content': message}
response = requests.post(discord_token, json=payload)
print(response)