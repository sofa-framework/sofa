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
template class RegularGridSpringForceField<Vec2dTypes>;
template class RegularGridSpringForceField<Vec2fTypes>;
template class RegularGridSpringForceField<Vec1dTypes>;
template class RegularGridSpringForceField<Vec1fTypes>;
template class RegularGridSpringForceField<Vec6dTypes>;
template class RegularGridSpringForceField<Vec6fTypes>;

// Register in the Factory
int RegularGridSpringForceFieldClass = core::RegisterObject("Spring acting on the edges and faces of a regular grid")
        .add< RegularGridSpringForceField<Vec3dTypes> >()
        .add< RegularGridSpringForceField<Vec3fTypes> >()
        .add< RegularGridSpringForceField<Vec2dTypes> >()
        .add< RegularGridSpringForceField<Vec2fTypes> >()
        .add< RegularGridSpringForceField<Vec1dTypes> >()
        .add< RegularGridSpringForceField<Vec1fTypes> >()
        .add< RegularGridSpringForceField<Vec6dTypes> >()
        .add< RegularGridSpringForceField<Vec6fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa

