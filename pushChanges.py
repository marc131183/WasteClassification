import subprocess

bashAddAll = "git add -A"
bashCommit = "git commit -m 'automatic push by jetson nano'"
bashPush = "git push"

subprocess.run(bashAddAll.split())
subprocess.run(bashCommit.split())
subprocess.run(bashPush.split())
