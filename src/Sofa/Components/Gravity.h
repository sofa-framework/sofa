#ifndef SOFA_COMPONENTS_GRAVITY_H
#define SOFA_COMPONENTS_GRAVITY_H

#include <Sofa/Components/Common/Vec.h>
#include <Sofa/Abstract/ContextObject.h>

namespace Sofa
{

namespace Components
{

namespace Graph
{
class GNode;
}

class Gravity : public Abstract::ContextObject
{
    typedef Common::Vec3d Vec3;
public:
    Gravity();
    const Vec3&  getGravity() const;

    /// Set the value of the gravity and return this
    Gravity* setGravity( const Vec3& g );

    /// Modify the context of the GNode
    void apply();
protected:
    Vec3 gravity_;
};

} // namespace Components

} // namespace Sofa

#endif

