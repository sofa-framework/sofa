#ifndef SOFA_COMPONENT_CONTEXTOBJECT_GRAVITY_H
#define SOFA_COMPONENT_CONTEXTOBJECT_GRAVITY_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/objectmodel/ContextObject.h>

namespace sofa
{

namespace simulation
{
namespace tree
{
class GNode;
}
}

namespace component
{

namespace contextobject
{

using namespace sofa::defaulttype;

/** Override the default gravity */
class Gravity : public core::objectmodel::ContextObject
{
    typedef defaulttype::Vec3d Vec3;
public:
    Gravity();

    DataField<Vec3> f_gravity; ///< Gravity in the world coordinate system

    /// Modify the context of the GNode
    void apply();
};

} // namespace contextobject

} // namespace component

} // namespace sofa

#endif

