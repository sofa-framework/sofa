SOFA_DIR=.
TEMPLATE = subdirs

include($$SOFA_DIR/sofa.cfg)

SUBDIRS += extlibs/NewMAT
SUBDIRS += extlibs/SLC
SUBDIRS += framework
SUBDIRS += modules
SUBDIRS += applications
