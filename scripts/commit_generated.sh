#!/usr/bin/env bash
# Commit and push a generated snapshot from a runner, and do not lose it.
#
# Two runs have now thrown away irreplaceable data at this exact step. Run
# 36274041202 pushed into a moved branch and lost that week's membership
# snapshot. Run 36285005849 fetched sectors for 1,220 symbols over seventeen
# minutes, hit a rebase conflict on the sidecar, and then retried `git pull
# --rebase` twice against an unmerged work tree -- which can only fail. The
# data was gone with the runner.
#
# The mistake both times was treating a generated file as an edit. It is not:
# the copy on the branch is an older generation of the same artifact, and this
# run holds the newer one. There is nothing to merge and no conflict to
# resolve. So on a rejected push this rebuilds the commit on top of the branch
# as it now stands and writes this run's files over it. The history keeps every
# older generation, which is the point -- for `registry/universe/us.csv` that
# history *is* the delisting record.
#
# Usage: scripts/commit_generated.sh <path> <commit message> [branch]
set -euo pipefail

path=$1
message=$2
branch=${3:-main}

if [ -z "$(git status --porcelain "$path")" ]; then
  echo "no change under $path"
  exit 0
fi

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

# Outside the work tree, so that a reset cannot take it.
keep=$(mktemp -d)
tar -cf "$keep/generated.tar" "$path"

for attempt in 1 2 3; do
  git add "$path"
  git commit -q -m "$message" || echo "nothing new to commit on attempt $attempt"
  if git push origin "HEAD:$branch"; then
    echo "pushed on attempt $attempt"
    exit 0
  fi
  echo "push attempt $attempt was rejected; rebuilding on top of origin/$branch" >&2
  git fetch origin "$branch"
  git reset -q --hard "origin/$branch"
  tar -xf "$keep/generated.tar"
  sleep $((attempt * 5))
done

echo "could not push $path after three attempts" >&2
exit 1
