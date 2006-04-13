#ifndef SOFA_COMPONENTS_GL_TEMPLATE_H
#define SOFA_COMPONENTS_GL_TEMPLATE_H

#include "../Common/config.h"
#include <GL/gl.h>

namespace Sofa
{

namespace Components
{

namespace GL
{

using namespace Common;

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

} // namespace GL

} // namespace Components

} // namespace Sofa

#endif
