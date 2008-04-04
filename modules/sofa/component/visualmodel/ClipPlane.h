#ifndef SOFA_COMPONENT_CLIPPLANE_H
#define SOFA_COMPONENT_CLIPPLANE_H

#include <sofa/core/VisualModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

using sofa::defaulttype::Vector3;

class ClipPlane : public core::VisualModel
{
public:
    Data<Vector3> position;
    Data<Vector3> normal;
    Data<int> id;
    Data<bool> active;

    ClipPlane();
    virtual ~ClipPlane();

    virtual void init();
    virtual void reinit();
    virtual void fwdDraw(Pass);
    virtual void bwdDraw(Pass);

protected:
    bool wasActive;
    double saveEq[4];
};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
