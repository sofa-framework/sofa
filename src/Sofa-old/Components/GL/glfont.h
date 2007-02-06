#ifndef SOFA_COMPONENTS_GL_GLFONT_H
#define SOFA_COMPONENTS_GL_GLFONT_H

namespace Sofa
{

namespace Components
{

namespace GL
{

void glfntInit(void);
void glfntWriteBitmap(float x,float y,char *s);
void glfntClose(void);

} // namespace GL

} // namespace Components

} // namespace Sofa

#endif
