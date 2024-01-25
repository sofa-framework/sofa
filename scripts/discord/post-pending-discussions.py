#!python


import os
import requests
from python_graphql_client import GraphqlClient

client = GraphqlClient(endpoint="https://api.github.com/graphql")
github_token = os.environ['GITHUB_TOKEN']
discord_token = os.environ['DISCORD_MAIN_WEBHOOK_URL']

# List of the repository to scan
repos=[['sofa-framework','sofa']]

# List of reviewers on GitHub Discussions
reviewer_logins=[["alxbilger"],["hugtalbot"],["bakpaul"],["fredroy"],["epernod"],["damienmarchal"],["VannesteFelix"],["EulalieCoevoet"],["adagolodjo"],["github-actions"]]


def computeListOfOpenDiscussionsPerCategory():
    for repo in repos:

        owner = repo[0]
        name = repo[1]

        has_next_page = True
        after_cursor = None

        categories = []
        discussions_numbers = []


        while has_next_page:
            # Trigger the query on discussions
            data = client.execute(
                query = make_query_discussions(owner, name, after_cursor),
                headers = {"Authorization": "Bearer {}".format(github_token)},
            )

            # Process each discussion
            for discussion in data["data"]["repository"]["discussions"]["nodes"]:

                # Exit if discussion is closed or answered
                if discussion["closed"] == True or discussion["isAnswered"] == True :
                    continue

                ##############################
                # Detect the last comment
                lastCommentId = len(discussion["comments"]["nodes"]) - 1

                # No comment at all
                if(lastCommentId < 0):
                    categories.append(discussion["category"]["name"])
                    discussions_numbers.append(discussion["number"])
                    continue

                lastReplyOnLastComment = len(discussion["comments"]["nodes"][lastCommentId]["replies"]["nodes"]) - 1

                # No replies on the last comment
                if(lastReplyOnLastComment < 0):
                    author = discussion["comments"]["nodes"][lastCommentId]["author"]["login"]
                # Select the last reply of the last comment
                else:
                    author = discussion["comments"]["nodes"][lastCommentId]["replies"]["nodes"][lastReplyOnLastComment]["author"]["login"]

                authorAsList = [author]

                # Check if author is indeed a reviewer
                if authorAsList in reviewer_logins:
                    continue
                else:
                    categories.append(discussion["category"]["name"])
                    discussions_numbers.append(discussion["number"])

            # save if request has another page to browse and its cursor pointers
            has_next_page = data["data"]["repository"]["discussions"]["pageInfo"]["hasNextPage"]
            after_cursor = data["data"]["repository"]["discussions"]["pageInfo"]["endCursor"]
    return categories, discussions_numbers


def printDiscussionsPerCategory(categories, discussions_numbers):
    Message = ":speech_balloon: GitHub pending discussion topics :speech_balloon: "
    postOnDiscord(Message)

    categoryDone = []

    for category in categories:
        tempVecID = []

        if category in categoryDone:
            continue

        for i,number in enumerate(discussionsNumbers):
            if categories[i] == category:
                tempVecID.append(number)
        tempMessage = "- Category "+str(category)+":"

        for id in tempVecID:
            tempMessage = tempMessage + " [#"+ str(id) +"](https://github.com/sofa-framework/sofa/discussions/"+ str(id) +") "

        # Category has been covered
        postOnDiscord(tempMessage)
        Message = Message + tempMessage + "\n"
        categoryDone.append(category)

    #print(Message)
    postOnDiscord(":fire: SOFA community appreciates all your support :fire: \n")

    return



# Function posting a message on Discord
def postOnDiscord(message):
    payload = {
        'content' : message,
        'username' : 'SOFA Github bot',
        'flags' : 4,
    }
    response = requests.post(discord_token, json=payload)
    print("Status: "+str(response.status_code)+"\nReason: "+str(response.reason)+"\nText: "+str(response.text))
    return



# Query to access all discussions
def make_query_discussions(owner, name, after_cursor=None):
    query = """
      query {
        repository(owner: "%s" name: "%s") {
          discussions(answered: false, first: 10, after:AFTER) {
            totalCount
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              number
              isAnswered
              closed
              category {
                name
              }
              comments (first: 100) {
                nodes {
                  author {
                    login
                  }
                  replies (first: 100) {
                    nodes {
                      author {
                        login
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }""" % (owner, name)
    return query.replace("AFTER", '"{}"'.format(after_cursor) if after_cursor else "null")



#==========================================================
# STEPS computed by the script
#==========================================================
# 1 - get the discussion to be warned and closed
result = computeListOfOpenDiscussionsPerCategory()
categories = result[0]
discussionsNumbers = result[1]

printDiscussionsPerCategory(categories, discussionsNumbers)
#==========================================================