#include <sofa/component/forcefield/SparseGridSpringForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

SOFA_DECL_CLASS(SparseGridSpringForceField)

using namespace sofa::defaulttype;

template class SparseGridSpringForceField<Vec3dTypes>;
template class SparseGridSpringForceField<Vec3fTypes>;

// Register in the Factory
int SparseGridSpringForceFieldClass = core::RegisterObject("Springs acting on the cells of a sparse grid")
        .add< SparseGridSpringForceField<Vec3dTypes> >()
        .add< SparseGridSpringForceField<Vec3fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa

