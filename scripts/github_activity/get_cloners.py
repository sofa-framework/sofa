import requests
import os
import sys

def get_clone_stats(owner, repo, auth_token):
    headers = {
        "Authorization": f"token {auth_token}"
    }
    url = f"https://api.github.com/repos/{owner}/{repo}/traffic/clones"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        unique_cloners = data['uniques']
        sum_unique_cloners = sum(stats['uniques'] for stats in data['clones'])
        total_cloners = data['count']
        sum_total_cloners = sum(stats['count'] for stats in data['clones'])
        #print(data)
        #print(sum_unique_cloners)
        return unique_cloners, total_cloners, sum_unique_cloners, sum_total_cloners
    else:
        print(f"Failed to fetch clone statistics. Error: {response.status_code}")

# Example usage
repo_owner = "sofa-framework"
repo_name = "sofa"
if "GITHUB_TOKEN" in os.environ:
    github_token = os.environ['GITHUB_TOKEN']
else:
    print("GITHUB_TOKEN environment variable is missing.")
    sys.exit(1)

unique_cloners, total_cloners, sum_unique_cloners, sum_total_cloners= get_clone_stats(repo_owner, repo_name, github_token)
print(f"Nombre de cloners uniques (API): {unique_cloners}") # to remove
print(f"Nombre total de cloners  (API): {total_cloners}") # to remove
print(f"Nombre de cloners uniques (sum): {sum_unique_cloners}")
print(f"Nombre total de cloners (sum): {sum_total_cloners}")
