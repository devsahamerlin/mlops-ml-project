# ML Project for MLOps

## Installation
```shell
python3 -m venv .venv
source .venv/bin/activate

touch .gitignore
vi .gitignore 

jupyter lab 

```

## Commandes
```shell

git init
git add .gitignore 
git commit -m "Create .gitignore file"
git log

git status
git add .
git commit -m "feat: add train, evaluate and config files"

git branch -M main
git remote add origin https://github.com/devsahamerlin/mlops-ml-project.git

git push -u origin main

git checkout -b dev
git push --set-upstream origin dev

git checkout -b feature/preprocessing
git add README.md
git commit -m "docs: add installation and git command in README file"

git checkout -b feature/evolutions
pip install -r requirements.txt

git add .
git commit -m "core: structure et installation des dépendances"
```

## Liste des artifacts