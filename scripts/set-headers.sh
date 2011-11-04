#!/bin/bash
for base in $(find framework modules applications -name copyright.txt | awk '{ print substr($0,0,length($0)-13) }' | sort); do
find $base \( -iname '*.h' -o -iname '*.cpp' -o -iname '*.inl' -o -iname '*.c' -o -iname '*.cu' \) -exec scripts/set-header.sh \{\} $base/copyright.txt \;
done
