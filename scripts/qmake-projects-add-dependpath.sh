#!/bin/bash
awk '$1 == "INCLUDEPATH" { print $0; $1 = "DEPENDPATH"; print $0; next; } { print } ' < "$1" > "$1".tmp
mv -f "$1".tmp "$1"

