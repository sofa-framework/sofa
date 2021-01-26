  #define SOFA_COMPONENT_FORCEFIELD_QUADBENDINGFEMFORCEFIELD_CPP

  #include "QuadBendingFEMForceField.inl"
  #include <sofa/core/ObjectFactory.h>
  #include <sofa/defaulttype/VecTypes.h>
  #include <cassert>

  namespace sofa
  {

  namespace component
  {

  namespace forcefield
  {

  using namespace sofa::defaulttype;

  // Register in the Factory
  int QuadBendingFEMForceFieldClass = core::RegisterObject("Bending Quad finite elements").add< QuadBendingFEMForceField<Vec3Types> >();

  template class SOFA_MISC_FEM_API QuadBendingFEMForceField<Vec3Types>;

  }

  }

  }
