#!python

import os, re, requests

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
PR_NUMBER = os.getenv('PR_NUMBER')
OWNER_NAME = os.getenv('OWNER_NAME')
PR_COMMIT_SHA = os.getenv('PR_COMMIT_SHA')


if (not GITHUB_TOKEN) or (not PR_NUMBER) or (not OWNER_NAME) or (not PR_COMMIT_SHA):
    print("Error: Missing required environment variables.")
    if (not GITHUB_TOKEN):
        print("     - Missing GITHUB_TOKEN")
    if (not PR_NUMBER):
        print("     - Missing PR_NUMBER")
    if (not OWNER_NAME):
        print("     - Missing OWNER_NAME")
    if (not PR_COMMIT_SHA):
        print("     - Missing PR_COMMIT_SHA")
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
dependency_dict = {}

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
    print(f"Labels found: {labels}.")

    if "pr: status to review" in labels:
        to_review_label_found = True
        print("PR is marked as 'to review'.")
    else:
        print(f"Flag to review has not been found. CI will stop.")
        exit(1)


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

    if any("[with-all-tests]" in comment for comment in comments):
        with_all_tests_found = True
        print("Found a comment containing 'with-all-tests'.")
    if any("[force-full-build]" in comment for comment in comments):
        force_full_build_found = True
        print("Found a comment containing 'with-all-tests'.")


# ========================================================================

# Export all needed PR information
def export_pr_info():
    pr_url = f"{API_URL}/pulls/{PR_NUMBER}"
    response = requests.get(pr_url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Failed to fetch pull request details: {response.status_code}")
        exit(1)

    pr_data = response.json()

    pr_url = str(pr_data['user']['html_url']) + "/" + str(pr_data['base']['repo']['name'])
    pr_branch_name = pr_data['head']['ref']
    pr_commit_sha = pr_data['head']['sha']

    print("PR comes from the repository: "+str(pr_url))
    print("PR branch name is: "+str(pr_branch_name))
    print("PR commit sha is: "+str(pr_commit_sha))

    with open(os.environ["GITHUB_ENV"], "a") as env_file:
        env_file.write(f"PR_OWNER_URL={pr_url}\n")
        env_file.write(f"PR_BRANCH_NAME={pr_branch_name}\n")
        env_file.write(f"PR_COMMIT_SHA={pr_commit_sha}\n")

    ## TODO : pr_data.get('mergeable', False) could also let us know if it is mergeable


# ========================================================================

# Extract repositories from ci-depends-on
def extract_ci_depends_on():
    global dependency_dict

    pr_url = f"{API_URL}/pulls/{PR_NUMBER}"
    response = requests.get(pr_url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Failed to fetch pull request details: {response.status_code}")
        exit(1)

    pr_data = response.json()

    # Extract the PR description and look for [ci-depends-on ...] patterns
    pr_body = pr_data.get("body", "")
    ci_depends_on = []

    # Search in each line for the pattern "[ci-depends-on ...]"
    for line in pr_body.splitlines():
        match = re.search(r'\[ci-depends-on (.+?)\]', line)
        if match:
            dependency = match.group(1).strip()
            ci_depends_on.append(dependency)
            print(f"Found ci-depends-on dependency: {dependency}")

            # Ensure the URL is in the expected dependency format, e.g. https://github.com/sofa-framework/Sofa.Qt/pull/6
            parts = dependency.split('/')
            if len(parts) != 7 or parts[0] != 'https:' or parts[1] != '' or parts[2] != 'github.com':
                raise ValueError("")
                print(f"Invalid URL ci-depends-on format: {dependency}")
                exit(1)

            owner = parts[3]
            repo = parts[4]
            pull_number = parts[6]
            dependency_request_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"

            response = requests.get(dependency_request_url, headers=HEADERS)

            if response.status_code != 200:
                print(f"Failed to fetch pull request details: {response.status_code}")
                exit(1)

            dependency_pr_data = response.json()

            key = dependency_pr_data['base']['repo']['name'] #Sofa.Qt
            repo_url = dependency_pr_data['head']['repo']['html_url'] #https://github.com/{remote from which pr comes}/Sofa.Qt
            branch_name = dependency_pr_data['head']['ref'] #my_feature_branch

            dependency_dict[key] = {
                "repo_url": repo_url,
                "branch_name": branch_name
            }
        
        match = re.search(r'\[with-all-tests\]', line)
        if match:
            with_all_tests_found = True


# ========================================================================
# Script core
# ========================================================================


# Execute the checks
check_labels()
check_if_draft()

# Trigger the build if conditions are met
if to_review_label_found and not is_draft_pr:
    # Export PR information (url, name, sha)
    export_pr_info()

    # Check compilation options in PR comments
    check_comments()
    
    # Extract dependency repositories
    extract_ci_depends_on()

    # Export all environment variables specific to pull-requests
    with open(os.environ["GITHUB_ENV"], "a") as env_file:
        env_file.write(f"WITH_ALL_TESTS={with_all_tests_found}\n")
        env_file.write(f"FORCE_FULL_BUILD={force_full_build_found}\n")

        ci_depends_on_str = f"{dependency_dict}".replace("'", "\\\"")
        env_file.write(f"CI_DEPENDS_ON={ci_depends_on_str}\n")
        env_file.write(f'BUILDER_OS=["sh-ubuntu_gcc_release","sh-fedora_clang_release","sh-windows_vs2022_release","sh-macos_clang_release"]')


# ========================================================================
