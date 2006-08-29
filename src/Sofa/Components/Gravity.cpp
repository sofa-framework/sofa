#include "Sofa/Components/Gravity.h"
#include "Sofa/Components/Common/Vec3Types.h"
#include <Sofa/Components/Graph/GNode.h>
#include "Sofa/Components/Common/ObjectFactory.h"

#include <math.h>

namespace Sofa
{

namespace Components
{


using namespace Common;
using namespace Core;

Gravity::Gravity()
    : Abstract::ContextObject()
{
}

const Gravity::Vec3&  Gravity::getGravity() const
{
    return gravity_;
}

Gravity* Gravity::setGravity( const Vec3& g )
{
    gravity_=g;
    return this;
}

void Gravity::apply()
{
    getContext()->setGravityInWorld( gravity_ );
}

void create(Gravity*& obj, ObjectDescription* arg)
{
    // TODO: read the parameters before
    obj = new Gravity;
    obj->setGravity(Vec3d(atof(arg->getAttribute("x","0")), atof(arg->getAttribute("y","0")), atof(arg->getAttribute("z","0"))));
}

SOFA_DECL_CLASS(Gravity)

Creator<ObjectFactory, Gravity> GravityClass("Gravity");

} // namespace Components

} // namespace Sofa

