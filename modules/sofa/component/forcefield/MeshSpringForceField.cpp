#include <sofa/component/forcefield/MeshSpringForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

template class MeshSpringForceField<Vec3dTypes>;
template class MeshSpringForceField<Vec3fTypes>;
template class MeshSpringForceField<Vec2dTypes>;
template class MeshSpringForceField<Vec2fTypes>;
template class MeshSpringForceField<Vec1dTypes>;
template class MeshSpringForceField<Vec1fTypes>;

SOFA_DECL_CLASS(MeshSpringForceField)

int MeshSpringForceFieldClass = core::RegisterObject("Spring force field acting along the edges of a mesh")
        .add< MeshSpringForceField<Vec3dTypes> >()
        .add< MeshSpringForceField<Vec3fTypes> >()
        .add< MeshSpringForceField<Vec2dTypes> >()
        .add< MeshSpringForceField<Vec2fTypes> >()
        .add< MeshSpringForceField<Vec1dTypes> >()
        .add< MeshSpringForceField<Vec1fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa

