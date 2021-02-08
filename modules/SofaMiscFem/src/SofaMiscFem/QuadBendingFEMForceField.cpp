/******************************************************************************
*						                                          *
*	     TouchIoT: Smart Tangible Sensing Enabler for Tactile Internet	*
*		              Developer: Nguyen Huu Nhan                          *
*                                  					                  *
******************************************************************************/
#define SOFA_COMPONENT_FORCEFIELD_QUADBENDINGFEMFORCEFIELD_CPP

#include "QuadBendingFEMForceField.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <cassert>

namespace sofa::component::forcefield
{

using namespace sofa::defaulttype;

// Register in the Factory
int QuadBendingFEMForceFieldClass = core::RegisterObject("Bending Quad finite elements")
      .add< QuadBendingFEMForceField<Vec3Types> >();

template class SOFA_SOFAMISCFEM_API QuadBendingFEMForceField<Vec3Types>;

} // namespace sofa::component::forcefield
