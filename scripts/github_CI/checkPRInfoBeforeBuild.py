#!python

import os
import requests
from python_graphql_client import GraphqlClient


client = GraphqlClient(endpoint="https://api.github.com/graphql")
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
PR_NUMBER = os.getenv('PR_NUMBER')
OWNER_NAME = os.getenv('OWNER_NAME')
COMMIT_SHA = os.getenv('COMMIT_SHA')


if not GITHUB_TOKEN or not PR_NUMBER or not REPO_NAME:
    print("Error: Missing required environment variables.")
    exit(1)


# GitHub API base URL
API_URL = f"https://api.github.com/repos/bakpaul/sofa"

# Headers for authentication
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}


# Flags to determine actions
to_review_label_found = False
is_draft_pr = False
with_all_tests_found = False
force_full_build_found = False


# ========================================================================

# Check PR labels
def check_labels():
    global to_review_label_found
    labels_url = f"{API_URL}/issues/{PR_NUMBER}/labels"
    response = requests.get(labels_url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Failed to fetch labels: {response.status_code}")
        exit(1)

    labels = [label['name'].lower() for label in response.json()]
    print(f"Labels found: {labels}")

    if "pr: status to review" in labels:
        to_review_label_found = True
        print("PR is marked as 'to review'.")

# ========================================================================

# Check the PR draft status
def check_if_draft():
    global is_draft_pr
    pr_url = f"{API_URL}/pulls/{PR_NUMBER}"
    response = requests.get(pr_url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Failed to fetch pull request details: {response.status_code}")
        exit(1)

    pr_data = response.json()
    is_draft_pr = pr_data.get('draft', False)

    if is_draft_pr:
        print("The pull request is a draft. The Bash script will not run.")

# ========================================================================

# Check PR comments for "[with-all-tests]" and "[force-full-build]"
def check_comments():
    global with_all_tests_found
    global force_full_build_found
    comments_url = f"{API_URL}/issues/{PR_NUMBER}/comments"
    response = requests.get(comments_url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Failed to fetch comments: {response.status_code}")
        exit(1)

    comments = [comment['body'].lower() for comment in response.json()]
    print(f"Comments found: {comments}")

    if any("[with-all-tests]" in comment for comment in comments):
        with_all_tests_found = True
        print("Found a comment containing 'with-all-tests'.")
    if any("[force-full-build]" in comment for comment in comments):
        force_full_build_found = True
        print("Found a comment containing 'with-all-tests'.")


# ========================================================================


# Execute the checks
check_labels()
check_if_draft()


# Trigger the build if conditions are met
if to_review_label_found and not is_draft_pr:
    
    # Check compilation options in PR comments
    check_comments()
    
    pr_url = f"{API_URL}/pulls/{PR_NUMBER}"
    response = requests.get(pr_url, headers=HEADERS)
    pr_data = response.json()
    branch = pr_data.get("head", {}).get("ref", None)
    
            
    # Extract the PR description and look for [ci-depends-on ...] patterns
    pr_body = pr_data.get("body", "")
    ci_depends_on = []

    for line in pr_body.splitlines():
        match = re.search(r'\[ci-depends-on (.+?)\]', line)
        if match:
            dependency = match.group(1).strip()
            ci_depends_on.append(dependency)
            print(f"Found ci-depends-on dependency: {dependency}")

    # GitHub repository details
    API_URL_DISPATCH = f"https://api.github.com/repos/bakpaul/sofa/dispatches"

    # JSON payload for the dispatch event
    PAYLOAD = {
        "event_type": "CI Build",
        "client_payload": {
            "owner": {OWNER_NAME},
            "branch": {branch},
            "commit_hash": {COMMIT_SHA},
            "preset": "full",
            "ci-depends-on": {ci_depends_on},
            "with-all-tests": {with_all_tests_found},
            "force-full-build": {force_full_build_found},
            "out-of-tree-build": "False",
            "generate-binaries": "False",
        },
    }

    # Headers for the GitHub API request
    REQUEST_HEADERS = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Perform the API request
    try:
        response = requests.post(API_URL_DISPATCH, json=PAYLOAD, headers=REQUEST_HEADERS)
        
        # Check for successful request
        if response.status_code == 204:
            print("CI Build event triggered successfully.")
        else:
            print(f"Failed to trigger CI Build event. Status code: {response.status_code}")
            print("Response:", response.text)
            sys.exit(1)
    except requests.RequestException as e:
        print("Error during the API request:", e)
        sys.exit(1)
else:
    print("Conditions not met. Bash script will not run.")

# ========================================================================
