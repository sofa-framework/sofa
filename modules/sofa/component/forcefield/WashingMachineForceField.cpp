#include <sofa/component/forcefield/WashingMachineForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

template class WashingMachineForceField<Vec3dTypes>;
template class WashingMachineForceField<Vec3fTypes>;

SOFA_DECL_CLASS(WashingMachineForceField)

// Register in the Factory
int WashingMachineForceFieldClass = core::RegisterObject("TODO")
        .add< WashingMachineForceField<Vec3dTypes> >()
        .add< WashingMachineForceField<Vec3fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa

