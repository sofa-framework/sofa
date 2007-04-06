#include <sofa/component/forcefield/BeamFEMForceField.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/core/ObjectFactory.h>
//#include <typeinfo>


namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

template class BeamFEMForceField<Rigid3dTypes>;


SOFA_DECL_CLASS(BeamFEMForceField)

// Register in the Factory
int BeamFEMForceFieldClass = core::RegisterObject("Beam finite elements")
        .add< BeamFEMForceField<Rigid3dTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa

