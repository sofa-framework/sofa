SOFA_DIR=../../..
TEMPLATE = subdirs

include($${SOFA_DIR}/sofa.cfg)

CONFIG += debug

SUBDIRS += lib
SUBDIRS += exe
 
