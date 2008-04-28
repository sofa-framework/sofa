#include <sofa/component/visualmodel/ClipPlane.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(ClipPlane)

int ClipPlaneClass = core::RegisterObject("OpenGL Clipping Plane")
        .add< ClipPlane >()
        ;


ClipPlane::ClipPlane()
    : position(initData(&position, Vector3(0,0,0), "position", "Point crossed by the clipping plane"))
    , normal(initData(&normal, Vector3(1,0,0), "normal", "Normal of the clipping plane, pointing toward the clipped region"))
    , id(initData(&id, 0, "id", "Clipping plane OpenGL ID"))
    , active(initData(&active,true,"active","Control whether the clipping plane should be applied or not"))
{

}

ClipPlane::~ClipPlane()
{
}

void ClipPlane::init()
{
}

void ClipPlane::reinit()
{
}

void ClipPlane::fwdDraw(Pass)
{
    wasActive = (bool)glIsEnabled(GL_CLIP_PLANE0+id.getValue());
    if (active.getValue())
    {
        glGetClipPlane(GL_CLIP_PLANE0+id.getValue(), saveEq);
        Vector3 p = position.getValue();
        Vector3 n = normal.getValue();
        GLdouble c[4] = { (GLdouble) -n[0], (GLdouble)-n[1], (GLdouble)-n[2], (GLdouble)(p*n) };
        glClipPlane(GL_CLIP_PLANE0+id.getValue(), c);
        if (!wasActive)
            glEnable(GL_CLIP_PLANE0+id.getValue());
    }
    else
    {
        if (wasActive)
            glDisable(GL_CLIP_PLANE0+id.getValue());
    }
}

void ClipPlane::bwdDraw(Pass)
{
    if (active.getValue())
    {
        glClipPlane(GL_CLIP_PLANE0+id.getValue(), saveEq);
        if (!wasActive)
            glDisable(GL_CLIP_PLANE0+id.getValue());
    }
    else
    {
        if (wasActive)
            glEnable(GL_CLIP_PLANE0+id.getValue());
    }
}

} // namespace visualmodel

} //namespace component

} //namespace sofa
