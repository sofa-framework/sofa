#include <sofa/component/forcefield/VectorSpringForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

template class VectorSpringForceField<Vec3dTypes>;
template class VectorSpringForceField<Vec3fTypes>;
//template class VectorSpringForceField<Vec2dTypes>;
//template class VectorSpringForceField<Vec2fTypes>;
//template class VectorSpringForceField<Vec1dTypes>;
//template class VectorSpringForceField<Vec1fTypes>;

SOFA_DECL_CLASS(VectorSpringForceField)

int VectorSpringForceFieldClass = core::RegisterObject("Spring force field acting along the edges of a mesh")
        .add< VectorSpringForceField<Vec3dTypes> >()
        .add< VectorSpringForceField<Vec3fTypes> >()
//.add< VectorSpringForceField<Vec2dTypes> >()
//.add< VectorSpringForceField<Vec2fTypes> >()
//.add< VectorSpringForceField<Vec1dTypes> >()
//.add< VectorSpringForceField<Vec1fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa
