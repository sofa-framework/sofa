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
    , f_gravity( dataField(&f_gravity,Vec3(0,0,0),"gravity","Gravity in the world coordinate system") )
{
}

// const Gravity::Vec3&  Gravity::getGravity() const
// {
//     return f_gravity.getValue();
// }
//
// Gravity* Gravity::setGravity( const Vec3& g )
// {
//     f_gravity.setValue(g);
//     return this;
// }

void Gravity::apply()
{
    getContext()->setGravityInWorld( f_gravity.getValue() );
}

void create(Gravity*& obj, ObjectDescription* arg)
{
    obj = new Gravity;
    obj->parseFields(arg->getAttributeMap() );
}

SOFA_DECL_CLASS(Gravity)

Creator<ObjectFactory, Gravity> GravityClass("Gravity");

} // namespace Components

} // namespace Sofa

