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

SOFA_DECL_CLASS(VectorSpringForceField)

int VectorSpringForceFieldClass = core::RegisterObject("Spring force field acting along the edges of a mesh")
#ifndef SOFA_FLOAT
        .add< VectorSpringForceField<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< VectorSpringForceField<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class VectorSpringForceField<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class VectorSpringForceField<Vec3fTypes>;
#endif


} // namespace forcefield

} // namespace component

} // namespace sofa
