#ifndef SOFA_HELPER_SYSTEM_GL_H
#define SOFA_HELPER_SYSTEM_GL_H
#include <sofa/helper/system/config.h>
#if defined (__APPLE__)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#ifdef WIN32
#include <GL/glext.h>
#endif
#endif
#endif
