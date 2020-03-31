#!/bin/bash

git pull
git add .
git status
read -p "Commit? [Y]/N: " flag
case $flag in
    [yY]|"$null")
        echo "yes"
        git commit -m "update"
        git push
        ;;
    *)
        echo "no"
        git reset HEAD
esac
git status