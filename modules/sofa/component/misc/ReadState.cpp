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

int ReadStateVec3fClass = core::RegisterObject("Read State vectors from file at each timestep")
#ifndef SOFA_FLOAT
        .add< ReadState<Vec3dTypes> >()
        .add< ReadState<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ReadState<Vec3fTypes> >()
        .add< ReadState<Rigid3fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class ReadState<Vec3dTypes>;
template class ReadState<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class ReadState<Vec3fTypes>;
template class ReadState<Rigid3fTypes>;
#endif
;

} // namespace misc

} // namespace component

} // namespace sofa
