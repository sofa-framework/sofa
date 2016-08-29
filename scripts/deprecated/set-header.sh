#!/bin/bash
echo ==== $1 - $2 ====
cp -p $1 $1.bak
if which dos2unix > /dev/null; then
if dos2unix -vt <$1 2>&1 | tail -1 | grep 'DOS line endings found.$' > /dev/null; then
unix2dos < $2 > $1;
else
cat $2 > $1;
fi
else
cat $2 > $1;
fi
awk ' BEGIN { header=0 } /^[ \t]*\/\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*/ { if (header==0) { header=1; next; } } /\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\// { if (header==1) { header=2; next; } } /^$/ { if (header==0) next; } { if (header==0) header=2; if (header!=1) print; } ' <$1.bak >> $1
rm -f $1.bak
