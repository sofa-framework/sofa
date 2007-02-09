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

SOFA_DECL_CLASS(MeshSpringForceField)

int MeshSpringForceFieldClass = core::RegisterObject("TODO")
        .add< MeshSpringForceField<Vec3dTypes> >()
        .add< MeshSpringForceField<Vec3fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa

