#ifndef SOFA_HELPER_SYSTEM_GLUT_H
#define SOFA_HELPER_SYSTEM_GLUT_H
#include <sofa/helper/system/gl.h>
#if defined (__APPLE__)
#include <GLUT/glut.h>
#else
// FIX compilation error with some glut headers combined with GLEW
#ifndef GLAPIENTRY
#define GLAPIENTRY
#endif
#include <GL/glut.h>
#endif
#endif
