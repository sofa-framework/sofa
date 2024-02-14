#!python

# DEPENDENCIES
# python3 -m pip install python-graphql-client

# REFERENCES
# How to request: https://stackoverflow.com/questions/42021113/how-to-use-curl-to-access-the-github-graphql-api
# Discussions API: https://docs.github.com/en/graphql/guides/using-the-graphql-api-for-discussions
# Pagination tuto: https://til.simonwillison.net/github/graphql-pagination-python

import os
import sys
import json
from datetime import datetime, timedelta, date
from python_graphql_client import GraphqlClient

client = GraphqlClient(endpoint="https://api.github.com/graphql")

if "GITHUB_TOKEN" in os.environ:
    github_token = os.environ['GITHUB_TOKEN']
else:
    print("GITHUB_TOKEN environment variable is missing.")
    sys.exit(1)


def make_query_discussions(owner, name, after_cursor=None):
    query = """
      query {
        repository(owner: "%s" name: "%s") {
          discussions(first: 10, after:AFTER) {
            totalCount
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              title
              authorAssociation
              createdAt
              author {
                login
              }
              answer {
                body
              }
              comments (first: 100) {
                nodes {
                  replies {
                    totalCount
                  }
                }
                totalCount
              }
            }
          }
        }
      }""" % (owner, name)
    return query.replace("AFTER", '"{}"'.format(after_cursor) if after_cursor else "null")

def make_query_issues(owner, name, after_cursor=None):
    query = """
      query {
        repository(owner: "%s" name: "%s") {
          issues(first: 10, after:AFTER) {
            totalCount
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              closed
              authorAssociation
              createdAt
              author {
                login
              }
            }
          }
        }
      }""" % (owner, name)
    return query.replace("AFTER", '"{}"'.format(after_cursor) if after_cursor else "null")

def make_query_PRs(owner, name, after_cursor=None):
    query = """
      query {
        repository(owner: "%s" name: "%s") {
          pullRequests(first: 10, after:AFTER) {
            totalCount
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              state
              merged
              authorAssociation
              createdAt
              author {
                login
              }
            }
          }
        }
      }""" % (owner, name)
    return query.replace("AFTER", '"{}"'.format(after_cursor) if after_cursor else "null")


def wasRecentlyCreated(createdAt):
  # Format date of creation YYYY-MM-DD
  creation_date = createdAt[:-10]
  creation_date = datetime.strptime(creation_date, '%Y-%m-%d')

  # Format the reference date (with which the created date will be compared)
  date_today = date.today()
  delay = timedelta(days = 31)
  date_reference = date_today - delay

  # Compare the creation date with reference date
  if creation_date.date() > date_reference:
    return True
  else :
    return False




# List of the repository to scan
repos=[['sofa-framework','sofa'],['SofaDefrost','SoftRobots']]


for repo in repos:

  owner = repo[0]
  name = repo[1]

  has_next_page = True
  after_cursor = None

  discussions = []
  replies_count = 0
  answers_count = 0
  recent_discussions_count = 0
  recent_discussions_authors = []

  issues = []
  issues_count = 0
  closed_count = 0
  recent_issues_count = 0
  recent_issues_authors = []

  pulls = []
  pulls_count = 0
  openPR_count = 0
  merged_count = 0
  pulls_first_timer_count = 0
  pulls_first_timers = []
  recent_pulls_count = 0
  recent_pulls_authors = []

  # DISCUSSIONS -----------------------------------------
  while has_next_page:
      data = client.execute(
          query = make_query_discussions(owner, name, after_cursor),
          headers = {"Authorization": "Bearer {}".format(github_token)},
      )

      # Get total number of discussions
      discussions = data["data"]["repository"]["discussions"]["totalCount"]

      # For each discussion
      for node in data["data"]["repository"]["discussions"]["nodes"]:
        
        # Count both the comments and associated replies
        for replies in node["comments"]["nodes"]:
          replies_count += replies["replies"]["totalCount"]

        replies_count += node["comments"]["totalCount"]

        # Count the discussions marked as "answered"
        if str(node["answer"]) != "None" :
          answers_count += 1

        # Filter all first timers, and make sure their first time was not double counted
        if wasRecentlyCreated(str(node["createdAt"])) == True:
          recent_discussions_count += 1
          recent_discussions_authors.append(node["author"]["login"])

      # save if request has another page to browse and its cursor pointers
      has_next_page = data["data"]["repository"]["discussions"]["pageInfo"]["hasNextPage"]
      after_cursor = data["data"]["repository"]["discussions"]["pageInfo"]["endCursor"]



  # ISSUES -----------------------------------------
  has_next_page = True
  after_cursor = None

  while has_next_page:
      data = client.execute(
          query = make_query_issues(owner, name, after_cursor),
          headers = {"Authorization": "Bearer {}".format(github_token)},
      )

      # Get total number of issues
      issues_count = data["data"]["repository"]["issues"]["totalCount"]

      # For each issue
      for node in data["data"]["repository"]["issues"]["nodes"]:
        # Count closed ones
        if str(node["closed"]) == "True" :
          closed_count += 1

        # Count recent and save authors
        if wasRecentlyCreated(str(node["createdAt"])) == True:
          recent_issues_count += 1
          recent_issues_authors.append(node["author"]["login"])

      has_next_page = data["data"]["repository"]["issues"]["pageInfo"]["hasNextPage"]
      after_cursor = data["data"]["repository"]["issues"]["pageInfo"]["endCursor"]



  # PRs -----------------------------------------
  has_next_page = True
  after_cursor = None

  while has_next_page:
      data = client.execute(
          query = make_query_PRs(owner, name, after_cursor),
          headers = {"Authorization": "Bearer {}".format(github_token)},
      )

      # Get total number of pull-requests
      pulls_count = data["data"]["repository"]["pullRequests"]["totalCount"]

      # For each PR
      for node in data["data"]["repository"]["pullRequests"]["nodes"]:
        # Count the open ones
        if str(node["state"]) == "OPEN" :
          openPR_count += 1
        # Count the merged ones
        if str(node["merged"]) == "True" :
          merged_count += 1

        # Count recent and save authors
        if wasRecentlyCreated(str(node["createdAt"])) == True:
          recent_pulls_count += 1
          recent_pulls_authors.append(node["author"]["login"])

          # Count first-time PR contributors
          if str(node["authorAssociation"]) == "FIRST_TIME_CONTRIBUTOR" :
            pulls_first_timer_count += 1
            pulls_first_timers.append(node["author"]["login"])

      has_next_page = data["data"]["repository"]["pullRequests"]["pageInfo"]["hasNextPage"]
      after_cursor = data["data"]["repository"]["pullRequests"]["pageInfo"]["endCursor"]


  # Filter array of authors keeping only unique names
  recent_discussions_authors = set(recent_discussions_authors)
  recent_issues_authors = set(recent_issues_authors)
  recent_pulls_authors = set(recent_pulls_authors)

  # Exporting values
  print("=========================")
  print("repo: "+str(owner)+", name: "+str(name))
  print("-------------------------")
  print(discussions, "discussion topics")
  print(answers_count, "answered")
  print(replies_count, "replies")
  # print(recent_discussions_count, "nb recent discussions")
  print(recent_discussions_authors, "recent discussion authors")
  print("-------------------------")
  print(issues_count, "issues")
  print(closed_count, "closed")
  # print(recent_issues_count, "nb recent issues")
  print(recent_issues_authors, "recent issue authors")
  print("-------------------------")
  print(pulls_count, "PRs")
  print(openPR_count, "open")
  print(merged_count, "merged")
  # print(recent_pulls_count, "nb recent PRs")
  print(recent_pulls_authors, "recent PR authors")
  print(pulls_first_timer_count, "nb pull 1st timers")
  print(pulls_first_timers, "1st timers")
  print("=========================")
  