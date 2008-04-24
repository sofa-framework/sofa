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


// Register in the Factory
int RegularGridSpringForceFieldClass = core::RegisterObject("Spring acting on the edges and faces of a regular grid")
#ifdef SOFA_FLOAT
        .add< RegularGridSpringForceField<Vec3fTypes> >(true) // default template
#else
        .add< RegularGridSpringForceField<Vec3dTypes> >(true) // default template
#ifndef SOFA_DOUBLE
        .add< RegularGridSpringForceField<Vec3fTypes> >() // default template
#endif
#endif
#ifndef SOFA_FLOAT
        .add< RegularGridSpringForceField<Vec2dTypes> >()
        .add< RegularGridSpringForceField<Vec1dTypes> >()
        .add< RegularGridSpringForceField<Vec6dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< RegularGridSpringForceField<Vec2fTypes> >()
        .add< RegularGridSpringForceField<Vec1fTypes> >()
        .add< RegularGridSpringForceField<Vec6fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class RegularGridSpringForceField<Vec3dTypes>;
template class RegularGridSpringForceField<Vec2dTypes>;
template class RegularGridSpringForceField<Vec1dTypes>;
template class RegularGridSpringForceField<Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class RegularGridSpringForceField<Vec3fTypes>;
template class RegularGridSpringForceField<Vec2fTypes>;
template class RegularGridSpringForceField<Vec1fTypes>;
template class RegularGridSpringForceField<Vec6fTypes>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

