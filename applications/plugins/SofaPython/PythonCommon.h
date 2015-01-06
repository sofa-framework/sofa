#ifndef SOFAPYTHON_PYTHON_H
#define SOFAPYTHON_PYTHON_H


// This header simply includes Python.h, taking care of platform-specific stuff

// It should be included before any standard headers:
// "Since Python may define some pre-processor definitions which affect the
// standard headers on some systems, you must include Python.h before any
// standard headers are included."

#if defined(_WIN32)
#	define MS_NO_COREDLL // deactivate pragma linking on Win32 done in Python.h
#	define Py_ENABLE_SHARED 1 // this flag ensure to use dll's version (needed because of MS_NO_COREDLL define).
#endif

#if defined(_MSC_VER) && defined(_DEBUG)
// if you use Python on windows in debug build, be sure to provide a compiled version because
// installation package doesn't come with debug libs.
#    include <Python.h> 
//#elif defined(__APPLE__) && defined(__MACH__)
//#    include <Python/Python.h>
#else
#    include <Python.h>
#endif

#endif
