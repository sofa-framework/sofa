#!/bin/bash
DIR0=$PWD
cd "${0%/*}"
SCRIPTS=$PWD
cd -
echo 'This script move all ExecParams/MechanicalParams/... in method prototypes and invokations as the first argument instead of the last in all source files within the given directories (or the current directory if none where specified).'
echo 'When the parameter is successfully moved, a /* PARAMS FIRST */ comment is added.'
echo 'If the auto-update fails, occurences of "@@@@" tags will remain and will have to be manually corrected.'
echo ''
echo 'WARNING: Your files will be modified. Use at your own risk !'
echo 'Press Enter to continue, or Ctrl+C to abort.'
read || exit 1
find ${@:-.} \( -iname '*.h' -o -iname '*.hpp' -o -iname '*.cpp' -o -iname '*.inl' -o -iname '*.c' \) -print -exec sed -i'~' -f $SCRIPTS/set-paramsfirst.sed '{}' ';'
