from git import Repo

repo = Repo(".")
repo.git.add(update=True)
repo.index.commit("automatic push from jetson nano")
origin = repo.remote(name="origin")
origin.push()
