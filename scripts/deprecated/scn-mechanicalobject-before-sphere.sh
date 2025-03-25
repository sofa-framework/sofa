#!/bin/bash
echo 'This script move all Sphere/SphereModel objects after MechanicalObjects in scene files in the current directory, as otherwise the SphereModel may not be successfully created, or not with the right template if the MechanicalObject in the parent node is of a different type.'
echo ''
echo 'WARNING: Your files will be modified. Use at your own risk !'
echo 'Press Enter to continue, or Ctrl+C to abort.'
read || exit 1

for f in $(find . -name '.svn' -prune -o -type f -print0 | xargs -0 grep -R -A 1 Sphere | grep -B 1 MechanicalObject | awk '{ if (substr($1, length($1), 1) == ":") print substr($1,1,length($1)-1); }' | uniq); do awk 'BEGIN { prev=""; } /Sphere/ { prev=$0; next; } /MechanicalObject/ { if (prev != "") { print $0; print prev; prev=""; next; } } { if (prev != "") { print prev; prev=""; } print; }' $f > $f.chg; mv -f $f.chg $f; done

echo ''
echo 'Here are remaining instances of Sphere as the first component within a Node (i.e. often meaning a MechanicalObject is missing):'
echo ''
find . -name '.svn' -prune -o -type f -print0 | xargs -0 grep -R -B 1 -E '(<T?Sphere|type="T?Sphere)' | grep -A 1 Node
