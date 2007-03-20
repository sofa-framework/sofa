#include "WriteState.inl"
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(WriteState)

using namespace defaulttype;

template class WriteState<Vec3dTypes>;
template class WriteState<Vec3fTypes>;
template class WriteState<RigidTypes>;

int WriteStateVec3fClass = core::RegisterObject("Write State vectors to file at each timestep")
        .add< WriteState<Vec3dTypes> >()
        .add< WriteState<Vec3fTypes> >()
        .add< WriteState<RigidTypes> >()
        ;

} // namespace misc

} // namespace component

} // namespace sofa
