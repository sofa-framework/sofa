#ifndef SOFA_HELPER_GL_GLFONT_H
#define SOFA_HELPER_GL_GLFONT_H

namespace sofa
{

namespace helper
{

namespace gl
{

void glfntInit(void);
void glfntWriteBitmap(float x,float y,char *s);
void glfntClose(void);

} // namespace gl

} // namespace helper

} // namespace sofa

#endif
