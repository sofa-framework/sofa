#!/bin/bash
# this script set the end of line style to native for all text files so that dump editors don't change all line endings and create big diffs hiding the real changes...
# Note that it is better to enable it as an auto-prop in subversion config: http://www.zope.org/DevHome/Subversion/SubversionConfigurationForLineEndings

echo dos2unix

find $* '(' -iname '*.c' -o -iname '*.cpp' -o -iname '*.cu' -o -iname '*.cxx'  -o -iname '*.h' -o -iname '*.hpp' -o -iname '*.inl' -o -iname '*.cuh' -o -iname '*.txt' -o -iname '*.pro' -o -iname '*.pri' -o -iname '*.prf' -o -iname '*.cfg' -o -iname '*.scn' -o -iname '*.xml' -o -iname '*.pscn' -o -iname '*.html' -o -iname '*.htm' -o -iname '*.php' ')' -exec /bin/bash -c 'file -m /dev/null "{}" | grep -q "CRLF, \(LF|CR\) line terminators$" && echo dos2unix "{}" && dos2unix "{}"' ';'

echo set svn:eol-style

find $* '(' -iname '*.c' -o -iname '*.cpp' -o -iname '*.cu' -o -iname '*.cxx'  -o -iname '*.h' -o -iname '*.hpp' -o -iname '*.inl' -o -iname '*.cuh' -o -iname '*.txt' -o -iname '*.pro' -o -iname '*.pri' -o -iname '*.prf' -o -iname '*.cfg' -o -iname '*.scn' -o -iname '*.xml' -o -iname '*.pscn' -o -iname '*.html' -o -iname '*.htm' -o -iname '*.php' ')' -exec /bin/bash -c 'P=`svn pg svn:eol-style "{}"` ; if [ "$P" == "" ]; then svn propset svn:eol-style native "{}" ; fi' ';'
