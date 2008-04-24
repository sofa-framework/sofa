
#include <sofa/component/misc/WriteState.inl>
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



int WriteStateVec3fClass = core::RegisterObject("Write State vectors to file at each timestep")
#ifndef SOFA_FLOAT
        .add< WriteState<Vec3dTypes> >()
        .add< WriteState<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< WriteState<Vec3fTypes> >()
        .add< WriteState<Rigid3fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class WriteState<Vec3dTypes>;
template class WriteState<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class WriteState<Vec3fTypes>;
template class WriteState<Rigid3fTypes>;
#endif

} // namespace misc

} // namespace component

} // namespace sofa
