#! /bin/sh

find . -name Makefile -exec rm '{}' ';'
qmake
