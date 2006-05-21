# message(Original config is $$CONFIG)
include(sofa.cfg)
SUBDIRS += src
TEMPLATE = subdirs
win32 {
TEMPLATE = $$TEMPLATESUBDIRS
}
# message(Final config is $$CONFIG)
