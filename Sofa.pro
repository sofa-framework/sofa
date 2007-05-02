SOFA_DIR=.
TEMPLATE = subdirs

include($$SOFA_DIR/sofa.cfg)

SUBDIRS += extlibs/NewMAT
SUBDIRS += extlibs/SLC
# PML
contains(DEFINES,SOFA_PML){
	SUBDIRS += extlibs/PML
	SUBDIRS += extlibs/LML
}

SUBDIRS += framework
SUBDIRS += modules
SUBDIRS += applications
