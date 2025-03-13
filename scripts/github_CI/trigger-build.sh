#!/bin/bash

# Check if exactly 8 arguments are provided
if [ "$#" -ne 9 ]; then
    echo "Error: Exactly 9 arguments are required."
    echo "Usage: $0 <GITHUB_TOKEN> <branch> <hash> <preset> <ci-depends-on> <with-all-tests> <force-full-build> <out-of-tree-build> <generate-binaries>"
    exit 1
fi

# Trigger the build action with all needed input:
# branch, hash, preset, ci-depends-on, with-all-tests, force-full-build, out-of-tree-build, label-wip/postponed, generate-binaries
sudo apt install curl
curl -L -X POST \
-H "Accept: application/vnd.github+json" \
-H "Authorization: Bearer $1"  \
-H "X-GitHub-Api-Version: 2022-11-28" \
https://api.github.com/repos/bakpaul/sofa/dispatches \
-d '{"event_type":"CI Build","client_payload":{"owner":"bakpaul","branch":"$2","commit_hash":"$3","preset":"$4","ci-depends-on":"$5", "with-all-tests":"$6", "force-full-build":"7", "out-of-tree-build":"$8", "generate-binaries":"$9"}}'
