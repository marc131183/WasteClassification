from git import Repo

def pushChanges():
	repo = Repo(".")
	repo.git.add(update=True)
	repo.index.commit("automatic push from jetson nano")
	origin = repo.remote(name="origin")
	origin.push()

if __name__ == "__main__":
	pushChanges()
