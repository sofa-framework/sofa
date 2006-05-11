#ifndef SOFA_COMPONENTS_GRAVITY_H
#define SOFA_COMPONENTS_GRAVITY_H

#include <Sofa/Core/Property.h>
#include <Sofa/Components/Common/Vec.h>

namespace Sofa
{

namespace Components
{
using Common::Vec3f;

class Gravity : public Core::Property
{
public:
    const Vec3f&  getGravity() const { return gravity_; }
    void setGravity( const Vec3f& g ) { gravity_=g; }

    void updateProperties( Core::Properties& data ) { data.gravity = gravity_; }
protected:
    Vec3f gravity_;
};

} // namespace Components

} // namespace Sofa

#endif


