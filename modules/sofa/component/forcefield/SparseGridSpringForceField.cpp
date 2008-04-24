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


// Register in the Factory
int SparseGridSpringForceFieldClass = core::RegisterObject("Springs acting on the cells of a sparse grid")
#ifndef SOFA_FLOAT
        .add< SparseGridSpringForceField<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< SparseGridSpringForceField<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SparseGridSpringForceField<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SparseGridSpringForceField<Vec3fTypes>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

