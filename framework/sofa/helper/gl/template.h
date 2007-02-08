#ifndef SOFA_HELPER_GL_TEMPLATE_H
#define SOFA_HELPER_GL_TEMPLATE_H

#include <sofa/helper/system/config.h>
#include <GL/gl.h>

namespace sofa
{

namespace helper
{

namespace gl
{

template<class Coord>
inline void glVertexT(const Coord& c)
{
    glVertex3d(c[0],c[1],c[2]);
}

template<>
inline void glVertexT<double>(const double& c)
{
    glVertex3d(c,0,0);
}

template<>
inline void glVertexT<float>(const float& c)
{
    glVertex3d(c,0,0);
}

} // namespace gl

} // namespace helper

} // namespace sofa

#endif
