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

class Gravity : public core::objectmodel::ContextObject
{
    typedef defaulttype::Vec3d Vec3;
public:
    Gravity();
    //virtual const char* getTypeName() const { return "Gravity"; }

    DataField<Vec3> f_gravity;

//         const Vec3&  getGravity() const;
// 	Gravity* setGravity( const Vec3& g );

    /// Modify the context of the GNode
    void apply();
protected:
    //Vec3 gravity_;
};

} // namespace contextobject

} // namespace component

} // namespace sofa

#endif

