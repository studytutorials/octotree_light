# Procedure for making part of supereight-srl public

This procedure will result in a commit with potentially thousands of changes,
but only requires resolving merge conflicts once per file.

1. Create a new branch in supereight-public and hard reset it to
   upstream/devel (`git reset --hard upstream/devel`).
2. Squash all the commits you want to port to supereight-public into 1. You can
   use `git rebase -i HEAD~X` to squash the last `X-1` commits. Set the commit
   message to `Upstream commits HASH1..HASH2` and the description to the commit
   messages of all squashed commits. This helps in knowing where to start
   porting from the next time.
3. Cherry-pick the combined commit into supereight-public/devel and fix any
   conflicts.
4. Amend the commit to remove any files that should not be made public.
5. Merge supereight-public/devel into supereight-public/master.

