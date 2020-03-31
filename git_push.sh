#!/bin/bash

echo ">>> git pull"
git pull
echo ">>> git add ."
git add .
echo ">>> git status"
git status
read -p "Commit? [Y]/N: " flag
case $flag in
    [yY]|"$null")
        echo "yes"
        echo ">>> git commit -m \"update\""
        git commit -m "update"
        echo ">>> git push"
        git push
        ;;
    *)
        echo "no"
        echo ">>> git reset HEAD"
        git reset HEAD
esac
echo ">>> git status"
git status
