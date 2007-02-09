#include <sofa/component/forcefield/RegularGridSpringForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

SOFA_DECL_CLASS(RegularGridSpringForceField)

using namespace sofa::defaulttype;

template class RegularGridSpringForceField<Vec3dTypes>;
template class RegularGridSpringForceField<Vec3fTypes>;

// Register in the Factory
int RegularGridSpringForceFieldClass = core::RegisterObject("TODO")
        .add< RegularGridSpringForceField<Vec3dTypes> >()
        .add< RegularGridSpringForceField<Vec3fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa

