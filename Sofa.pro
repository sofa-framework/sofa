SOFA_DIR=.
TEMPLATE = subdirs

include($$SOFA_DIR/sofa.cfg)

SUBDIRS += Tools/NewMAT
SUBDIRS += framework
SUBDIRS += modules
SUBDIRS += applications
