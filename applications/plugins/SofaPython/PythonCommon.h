#ifndef SOFAPYTHON_PYTHON_H
#define SOFAPYTHON_PYTHON_H


// This header simply includes Python.h, taking care of platform-specific stuff

// It should be included before any standard headers:
// "Since Python may define some pre-processor definitions which affect the
// standard headers on some systems, you must include Python.h before any
// standard headers are included."
#if defined(_MSC_VER)
// intrusive_ptr.hpp has to be ahead of python.h on windows to support debug
// compilation.
#include <sofa/core/sptr.h>

// undefine _DEBUG since we want to always link to the release version of
// python and pyconfig.h automatically links debug version if _DEBUG is
// defined.
#ifdef _DEBUG
#define _DEBUG_UNDEFED
#undef _DEBUG
#endif

#endif


#include <Python.h>


#if defined(_MSC_VER)
// redefine _DEBUG if it was undefed
#ifdef _DEBUG_UNDEFED
#define _DEBUG
#endif
#endif

#endif
