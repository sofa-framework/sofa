SOFA_DIR=../../..
TEMPLATE = subdirs

include($${SOFA_DIR}/sofa.cfg)
CONFIG -= ordered

SUBDIRS += pim
SUBDIRS += gui
