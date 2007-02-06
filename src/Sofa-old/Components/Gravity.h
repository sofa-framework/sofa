#ifndef SOFA_COMPONENTS_GRAVITY_H
#define SOFA_COMPONENTS_GRAVITY_H

#include <Sofa-old/Components/Common/Vec.h>
#include <Sofa-old/Abstract/ContextObject.h>

namespace Sofa
{

namespace Components
{

namespace Graph
{
class GNode;
}

using namespace Common;

class Gravity : public Abstract::ContextObject
{
    typedef Common::Vec3d Vec3;
public:
    Gravity();
    virtual const char* getTypeName() const { return "Gravity"; }

    DataField<Vec3> f_gravity;

//         const Vec3&  getGravity() const;
// 	Gravity* setGravity( const Vec3& g );

    /// Modify the context of the GNode
    void apply();
protected:
    //Vec3 gravity_;
};

} // namespace Components

} // namespace Sofa

#endif

