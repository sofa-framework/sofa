#!/bin/bash
# this script set the end of line style to native for all text files so that dump editors don't change all line endings and create big diffs hiding the real changes...
# Note that it is better to enable it as an auto-prop in subversion config: http://www.zope.org/DevHome/Subversion/SubversionConfigurationForLineEndings
find $* '(' -name '*.c' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cxx'  -o -name '*.h' -o -name '*.hpp' -o -name '*.inl' -o -name '*.cuh' -o -name '*.txt' -o -name '*.pro' -o -name '*.pri' -o -name '*.cfg' ')' -exec svn propset svn:eol-style native '{}' ';'
