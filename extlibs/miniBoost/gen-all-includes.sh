#!/bin/bash
echo Processing excludes.txt
PATTERN="(^excluded"
for f in $(grep -v '^#' excludes.txt); do
PATTERN="$PATTERN|^$f";
done
PATTERN="$PATTERN)";
echo "$PATTERN"
echo Processing includes.txt
grep -v '^#' includes.txt | sort | uniq | grep -v "$PATTERN" > all-includes.txt
let it=1
while true; do
echo   iteration $it: $(wc -l < all-includes.txt) files;
for f in $(cat all-includes.txt); do if [ -f "$f" ]; then echo $f; fi; grep -oE '^#[ \t]*include[ \t]*[<"](boost/[a-zA-Z0-9_/.-]+)[">]' $f | sed 's/^.*[<"]\([^">]*\)[">]/\1/'; done | sort | uniq | grep -vE "$PATTERN" > new-all-includes.txt
let it+=1
if [ $(wc -l < all-includes.txt) -eq $(wc -l < new-all-includes.txt) ]; then
rm -f new-all-includes.txt
break
else
mv -f new-all-includes.txt all-includes.txt
fi
done
echo   iteration $it: $(wc -l < all-includes.txt) files;
echo Processing directories.txt
(cat all-includes.txt ; find $(grep -v '^#' directories.txt) -name .svn -prune -o -type f -print) | sort | uniq | grep -vE "$PATTERN" > new-all-includes.txt
mv -f new-all-includes.txt all-includes.txt
let it=1
while true; do
echo   iteration $it: $(wc -l < all-includes.txt) files;
for f in $(cat all-includes.txt); do if [ -f "$f" ]; then echo $f; fi; grep -oE '^#[ \t]*include[ \t]*[<"](boost/[a-zA-Z0-9_/.-]+)[">]' $f | sed 's/^.*[<"]\([^">]*\)[">]/\1/'; done | sort | uniq | grep -vE "$PATTERN" > new-all-includes.txt
let it+=1
if [ $(wc -l < all-includes.txt) -eq $(wc -l < new-all-includes.txt) ]; then
rm -f new-all-includes.txt
break
else
mv -f new-all-includes.txt all-includes.txt
fi
done
echo   iteration $it: $(wc -l < all-includes.txt) files;
