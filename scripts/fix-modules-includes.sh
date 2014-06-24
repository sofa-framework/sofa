#!/bin/sh

# Use this script to fix the includes broken by the transformation of modules into plugins

if [ "$#" -eq 0 ]; then
    echo "You really should provide an argument, e.g."
    echo "./scripts/fix-modules-includes.sh applications/plugins/MyPlugin"
    exit 1
fi

find $* "(" -iname "*.h" -o -iname "*.cpp" -o -iname "*.inl" -o -iname "*.txt" ")" -exec sed -i -f scripts/fix-modules-includes.sed {} \; 
