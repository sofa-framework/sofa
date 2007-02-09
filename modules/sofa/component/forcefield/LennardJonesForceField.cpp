#include <sofa/component/forcefield/LennardJonesForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{


namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace core::componentmodel::behavior;

// Each instance of our class must be compiled
template class LennardJonesForceField<Vec3fTypes>;
template class LennardJonesForceField<Vec3dTypes>;

SOFA_DECL_CLASS(LennardJonesForceField)

int LennardJonesForceFieldClass = core::RegisterObject("Lennard-Jones forces for fluids")
        .add< LennardJonesForceField<Vec3dTypes> >()
        .add< LennardJonesForceField<Vec3fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa

