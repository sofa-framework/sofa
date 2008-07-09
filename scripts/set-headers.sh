#!/bin/bash
find framework \( -iname '*.h' -o -iname '*.cpp' -o -iname '*.inl' -o -iname '*.c' -o -iname '*.cu' \) -exec scripts/set-header.sh \{\} framework/copyright.txt \;
find modules \( -iname '*.h' -o -iname '*.cpp' -o -iname '*.inl' -o -iname '*.c' -o -iname '*.cu' \) -exec scripts/set-header.sh \{\} modules/copyright.txt \;
find applications \( -iname '*.h' -o -iname '*.cpp' -o -iname '*.inl' -o -iname '*.c' -o -iname '*.cu' \) -exec scripts/set-header.sh \{\} applications/copyright.txt \;
