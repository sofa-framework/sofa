#include <sofa/component/forcefield/TetrahedronFEMForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
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

template class TetrahedronFEMForceField<Vec3dTypes>;
template class TetrahedronFEMForceField<Vec3fTypes>;


SOFA_DECL_CLASS(TetrahedronFEMForceField)

// Register in the Factory
int TetrahedronFEMForceFieldClass = core::RegisterObject("Tetrahedral finite elements")
        .add< TetrahedronFEMForceField<Vec3dTypes> >()
        .add< TetrahedronFEMForceField<Vec3fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa

