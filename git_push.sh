#!/bin/bash

# echo ">>> git pull origin master"
# git pull origin master
echo ">>> git pull gitee master"
git pull gitee master
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
        echo ">>> git push origin master"
        git push origin master
        echo ">>> git push gitee master"
        git push gitee master
        ;;
    *)
        echo "no"
        echo ">>> git reset HEAD"
        git reset HEAD
esac
echo ">>> git status"
git status
