#include <sofa/component/interactionforcefield/RepulsiveSpringForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(RepulsiveSpringForceField)

template class RepulsiveSpringForceField<Vec3dTypes>;
template class RepulsiveSpringForceField<Vec3fTypes>;

// Register in the Factory
int RepulsiveSpringForceFieldClass = core::RegisterObject("TODO")
        .add< RepulsiveSpringForceField<Vec3dTypes> >()
        .add< RepulsiveSpringForceField<Vec3fTypes> >()
        ;

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

