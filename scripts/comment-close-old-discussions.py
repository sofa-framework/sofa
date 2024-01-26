#!python

# DEPENDENCIES
# python3 -m pip install python-graphql-client

#Ref : Action in python https://www.python-engineer.com/posts/run-python-github-actions/


#TEST steps:
# - after 1st of December 2023, remove the temporary layer

import os
import sys
import json
from datetime import datetime, timedelta, date
from python_graphql_client import GraphqlClient
from dateutil.relativedelta import relativedelta


client = GraphqlClient(endpoint="https://api.github.com/graphql")
github_token = os.environ['GITHUB_TOKEN']


# List of the repository to scan
repos=[['sofa-framework','sofa']]


# Format the reference date (with which the last reply will be compared)
# Today
date_today = date.today()
# warning delay = 2-month delay for warning
delay_warning = relativedelta(months = 2)
date_reference_warning = date_today - delay_warning
# closing delay = 2+2.5-month delay for closing
delay_closing = relativedelta(months = 2) + relativedelta(weeks = 1)
date_reference_closing = date_today - delay_closing


# List of reviewers on GitHub Discussions
reviewer_logins=[["alxbilger"],["hugtalbot"],["bakpaul"],["fredroy"],["epernod"],["damienmarchal"],["VannesteFelix"],["adagolodjo"],["EulalieCoevoet"],["github-actions"]]


# Check if the "createdAt" is older than the "date_reference"
def isOlderThan(date_reference, createdAt):
  # Format date of creation YYYY-MM-DD
  creation_date = createdAt[:-10]
  creation_date = datetime.strptime(creation_date, '%Y-%m-%d')

  if creation_date.date() > date_reference:
    return False
  else :
    return True

# Returns true of the date "createdAt" is more than the warning delay
def isToBeWarned(createdAt):
  return isOlderThan(date_reference_warning, createdAt)


# Returns true of the date "createdAt" is more than the closing delay
def isToBeClosed(createdAt):
  return isOlderThan(date_reference_closing, createdAt)



def computeListOfDiscussionToProcess():
  for repo in repos:

    owner = repo[0]
    name = repo[1]

    has_next_page = True
    after_cursor = None

    to_be_warned_discussion_number = []
    to_be_warned_discussion_id = []
    to_be_warned_discussion_author = []

    to_be_closed_discussion_number = []
    to_be_closed_discussion_id = []
    to_be_closed_discussion_author = []

    while has_next_page:
        # Trigger the query on discussions
        data = client.execute(
            query = make_query_discussions(owner, name, after_cursor),
            headers = {"Authorization": "Bearer {}".format(github_token)},
        )

        # Process each discussion
        for discussion in data["data"]["repository"]["discussions"]["nodes"]:

          # Save original author of the discussion
          discussionAuthor = discussion["author"]["login"]

          # Detect the last comment
          lastCommentId = len(discussion["comments"]["nodes"]) - 1

          # Pass to the next discussion item if :
          # no comment in the discussion OR discussion is answered OR closed
          if(lastCommentId < 0 or discussion["closed"] == True or discussion["isAnswered"] == True ):
            continue

          lastReplyOnLastComment = len(discussion["comments"]["nodes"][lastCommentId]["replies"]["nodes"]) - 1

          # No replies on the last comment
          if(lastReplyOnLastComment < 0):
            author = discussion["comments"]["nodes"][lastCommentId]["author"]["login"]
            dateLastMessage = discussion["comments"]["nodes"][lastCommentId]["createdAt"]
          # Select the last reply of the last comment
          else:
            author = discussion["comments"]["nodes"][lastCommentId]["replies"]["nodes"][lastReplyOnLastComment]["author"]["login"]
            dateLastMessage = discussion["comments"]["nodes"][lastCommentId]["replies"]["nodes"][lastReplyOnLastComment]["createdAt"]
          
          authorAsList = [author]

          # Check if author is indeed a reviewer
          if authorAsList in reviewer_logins:
            #Check dates
            if isToBeClosed(dateLastMessage) == True:
              to_be_closed_discussion_number.append(discussion["number"])
              to_be_closed_discussion_id.append(discussion["id"])
              to_be_closed_discussion_author.append(discussionAuthor)
            elif isToBeWarned(dateLastMessage) == True   and   author != "github-actions":
              to_be_warned_discussion_number.append(discussion["number"])
              to_be_warned_discussion_id.append(discussion["id"])
              to_be_warned_discussion_author.append(discussionAuthor)


        # save if request has another page to browse and its cursor pointers
        has_next_page = data["data"]["repository"]["discussions"]["pageInfo"]["hasNextPage"]
        after_cursor = data["data"]["repository"]["discussions"]["pageInfo"]["endCursor"]
  return [to_be_warned_discussion_number,to_be_warned_discussion_id,to_be_warned_discussion_author,to_be_closed_discussion_number,to_be_closed_discussion_id,to_be_closed_discussion_author]


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
              id
              number
              isAnswered
              closed
              author {
                login
              }
              comments (first: 100) {
                nodes {
                  createdAt
                  author {
                    login
                  }
                  replies (first: 100) {
                    nodes {
                      createdAt
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




def make_github_warning_comment(discussion_id, discussion_author):
    message = ":warning: :warning: :warning:<br>@"+str(discussion_author)+"<br>Feedback has been given to you by the project reviewers, however we have not received a response from you. Without further news in the coming weeks, this discussion will be automatically closed in order to keep this forum clean and fresh :seedling: Thank you for your understanding"

    query = """
      mutation {
        addDiscussionComment(input: {body: "%s", discussionId: "%s"}) {
          comment {
            id
          }
        }
      }
""" % (message, discussion_id)
    return query



def make_github_closing_comment(discussion_id, discussion_author):
  message = ":warning: :warning: :warning:<br>@"+str(discussion_author)+"<br>In accordance with our forum management policy, the last reply is more than 4 months old and is therefore closed. Our objective is to keep the forum up to date and offer the best support experience.<br><br>Please feel free to reopen it if the topic is still active by providing us with an update. Please feel free to open a new thread at any time - we'll be happy to help where and when we can."

  query = """
    mutation {
      addDiscussionComment(input: {body: "%s", discussionId: "%s"}) {
        comment {
          id
        }
      }
    }
""" % (message, discussion_id)
  return query

def close_github_discussion(discussion_id):

  query = """
    mutation {
      closeDiscussion(input: {discussionId: "%s"}) {
        discussion {
          id
        }
      }
    }
""" % (discussion_id)
  return


# ------- TO REMOVE !!!! ---------
def temporary_github_closing_comment(discussion_id, discussion_author):
  message = ":warning: :warning: :warning:<br>@"+str(discussion_author)+"<br>We are setting a new forum management policy: topics like this one, in which the last reply is more than 4 months old, will automatically be closed. Without further news in the coming weeks, our bot will proceed to automatic closure, thus keeping this forum clean and fresh :seedling: Thank you for your understanding"

  query = """
    mutation {
      addDiscussionComment(input: {body: "%s", discussionId: "%s"}) {
        comment {
          id
        }
      }
    }
""" % (message, discussion_id)
  return query
# --------------------------------




#==========================================================
# STEPS computed by the script
#==========================================================
# 1 - get the discussion to be warned and closed
result = computeListOfDiscussionToProcess()
to_be_warned_discussion_number = result[0]
to_be_warned_discussion_id = result[1]
to_be_warned_discussion_author = result[2]
to_be_closed_discussion_number = result[3]
to_be_closed_discussion_id = result[4]
to_be_closed_discussion_author = result[5]
#==========================================================
# 2- do it using github API
if(len(to_be_warned_discussion_id)!=len(to_be_warned_discussion_author)):
  print('Error: size of both vectors number/author for discussions to be warned is different')
  exit(1)
if(len(to_be_closed_discussion_id)!=len(to_be_closed_discussion_author)):
  print('Error: size of both vectors number/author for discussions to be closed is different')
  exit(1)

print("** Output lists **")
print("******************")
print("Nb discussions to be WARNED = "+str(len(to_be_warned_discussion_number)))
print("Nb discussions to be CLOSED = "+str(len(to_be_closed_discussion_number)))
print("******************")
print("to_be_warned_discussion_number = "+str(to_be_warned_discussion_number))
print("to_be_warned_discussion_id = "+str(to_be_warned_discussion_id))
print("to_be_warned_discussion_author = "+str(to_be_warned_discussion_author))
print("******************")
print("to_be_closed_discussion_number = "+str(to_be_closed_discussion_number))
print("to_be_closed_discussion_id = "+str(to_be_closed_discussion_id))
print("to_be_closed_discussion_author = "+str(to_be_closed_discussion_author))
print("******************")
print("******************")

#==========================================================
# WARNING step
print("** WARNING step **")
for index, discussion_id in enumerate(to_be_warned_discussion_id):
  print("to_be_warned_discussion_number[index] = "+str(to_be_warned_discussion_number[index]))
  print("to_be_warned_discussion_author[index] = "+str(to_be_warned_discussion_author[index]))
  print("discussion_id = "+str(discussion_id))
  # Warning comment
  data = client.execute(
            query = make_github_warning_comment( discussion_id, to_be_warned_discussion_author[index] ),
            headers = {"Authorization": "Bearer {}".format(github_token)},
        )
  print(data)

print("******************")
print("******************")

#==========================================================
# CLOSING step
print("** CLOSING step **")

# ------- TO REMOVE !!!! ---------
date_today = date.today()
date_end_temporary_message = date.fromisoformat('2024-01-01')
temporary_case = False

if date_today > date_end_temporary_message:
    temporary_case = False
else:
    temporary_case = True

if temporary_case:
  remaining_time = date_end_temporary_message-date_today
  print(str(remaining_time)[:-9]+" days to go before end of temporary message")
# --------------------------------

for index, discussion_id in enumerate(to_be_closed_discussion_id):
  print("to_be_closed_discussion_number[index] = "+str(to_be_closed_discussion_number[index]))
  print("to_be_closed_discussion_author[index] = "+str(to_be_closed_discussion_author[index]))
  print("discussion_id = "+str(discussion_id))
  # ------- TO REMOVE !!!! ---------
  if temporary_case:
    # Closing comment
    data = client.execute(
             query = temporary_github_closing_comment( discussion_id, to_be_closed_discussion_author[index] ),
             headers = {"Authorization": "Bearer {}".format(github_token)},
         )
    print(data)
  else:
  # --------------------------------
    # Closing comment
    data = client.execute(
             query = make_github_closing_comment( discussion_id, to_be_closed_discussion_author[index] ),
             headers = {"Authorization": "Bearer {}".format(github_token)},
         )
    print(data)
    # Close discussion
    data = client.execute(
             query = close_github_discussion( discussion_id ),
             headers = {"Authorization": "Bearer {}".format(github_token)},
         )
    print(data)

#==========================================================