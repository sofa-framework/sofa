#include <sofa/component/contextobject/Gravity.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>
#include <math.h>


namespace sofa
{

namespace component
{

namespace contextobject
{

using namespace sofa::defaulttype;
using namespace core::componentmodel::behavior;

Gravity::Gravity()
    : core::objectmodel::ContextObject()
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

void create(Gravity*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    obj = new Gravity;
    obj->parseFields(arg->getAttributeMap() );
}

SOFA_DECL_CLASS(Gravity)

Creator<simulation::tree::xml::ObjectFactory, Gravity> GravityClass("Gravity");

} // namespace contextobject

} // namespace component

} // namespace sofa

