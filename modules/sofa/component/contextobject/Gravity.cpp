#include <sofa/component/contextobject/Gravity.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/core/ObjectFactory.h>
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
    : f_gravity( dataField(&f_gravity,Vec3(0,0,0),"gravity","Gravity in the world coordinate system") )
{
}

void Gravity::apply()
{
    getContext()->setGravityInWorld( f_gravity.getValue() );
}

SOFA_DECL_CLASS(Gravity)

int GravityClass = core::RegisterObject("Gravity in world coordinates")
        .add< Gravity >()
        ;

} // namespace contextobject

} // namespace component

} // namespace sofa

