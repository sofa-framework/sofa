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



SOFA_DECL_CLASS(BeamFEMForceField)

// Register in the Factory
int BeamFEMForceFieldClass = core::RegisterObject("Beam finite elements")
#ifndef SOFA_FLOAT
        .add< BeamFEMForceField<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< BeamFEMForceField<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class BeamFEMForceField<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class BeamFEMForceField<Rigid3fTypes>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

