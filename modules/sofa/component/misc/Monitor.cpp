/****************************************************************************
*																			*
*		Copyright: See COPYING file that comes with this distribution		*
*																			*
****************************************************************************/
#include <sofa/component/misc/Monitor.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(Monitor)

using namespace sofa::defaulttype;

// Register in the Factory
int MonitorClass = core::RegisterObject("Monitoring of particles")

#ifndef SOFA_FLOAT
        .add< Monitor<Vec3dTypes> >()
        .add< Monitor<Vec2dTypes> >()
        .add< Monitor<Vec1dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< Monitor<Vec3fTypes> >()
        .add< Monitor<Vec2fTypes> >()
        .add< Monitor<Vec1fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class Monitor<Vec3dTypes>;
template class Monitor<Vec2dTypes>;
template class Monitor<Vec1dTypes>;
//template class Monitor<Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class Monitor<Vec3fTypes>;
template class Monitor<Vec2fTypes>;
template class Monitor<Vec1fTypes>;
//template class Monitor<Vec6fTypes>;
#endif


} // namespace misc

} // namespace component

} // namespace sofa
