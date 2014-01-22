#ifndef SOFAPYTHON_PYTHON_H
#define SOFAPYTHON_PYTHON_H


// This header simply includes Python.h, taking care of platform-specific stuff

// It should be included before any standard headers:
// "Since Python may define some pre-processor definitions which affect the
// standard headers on some systems, you must include Python.h before any
// standard headers are included."


#if defined(_MSC_VER) && defined(_DEBUG)
// undefine _DEBUG since we want to always link agains the release version of
// python and pyconfig.h automatically links debug version if _DEBUG is defined.
#    undef _DEBUG
#    include <Python.h>
#    define _DEBUG
#elif defined(__APPLE__) && defined(__MACH__)
#    include <Python/Python.h>
#else
#    include <Python.h>
#endif


#endif
