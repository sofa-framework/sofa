#include <sofa/component/misc/ReadState.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(ReadState)

using namespace defaulttype;

template class ReadState<Vec3dTypes>;
template class ReadState<Vec3fTypes>;
template class ReadState<RigidTypes>;

int ReadStateVec3fClass = core::RegisterObject("Read State vectors from file at each timestep")
        .add< ReadState<Vec3dTypes> >()
        .add< ReadState<Vec3fTypes> >()
        .add< ReadState<RigidTypes> >()
        ;

} // namespace misc

} // namespace component

} // namespace sofa
