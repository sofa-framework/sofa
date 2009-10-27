#! /bin/sh

find . -name .svn -prune -o '-(' -name Makefile -o -name 'Makefile.*' '-)' -exec rm '{}' ';'
qmake -recursive
