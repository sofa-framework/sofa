#define SOFA_COMPONENT_FORCEFIELD_TORSIONFORCEFIELD_CPP

#include <sofa/core/ObjectFactory.h>
#include <sofa/component/forcefield/TorsionForceField.inl>

namespace sofa
{

namespace component
{

namespace forcefield
{


int TorsionForceFieldClass = core::RegisterObject("Applies a torque to specified points")
#ifndef SOFA_DOUBLE
		.add< TorsionForceField<Vec3fTypes> >()
		.add< TorsionForceField<Rigid3fTypes> >()
#endif
#ifndef SOFA_FLOAT
		.add< TorsionForceField<Vec3dTypes> >()
		.add< TorsionForceField<Rigid3dTypes> >()
#endif
;

#ifndef SOFA_FLOAT
template class TorsionForceField<Vec3dTypes>;
template class TorsionForceField<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class TorsionForceField<Vec3fTypes>;
template class TorsionForceField<Rigid3fTypes>;
#endif

} // namespace forcefield
} // namespace component
} // namespace sofa
