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


#ifdef SOFA_FLOAT
        .add< Monitor<Vec3fTypes> >(true) // default template
#else
        .add< Monitor<Vec3dTypes> >(true) // default template
#ifndef SOFA_DOUBLE
        .add< Monitor<Vec3fTypes> >() // default template
#endif
#endif
        ;

#ifndef SOFA_FLOAT
template class Monitor<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class Monitor<Vec3fTypes>;
#endif


} // namespace misc

} // namespace component

} // namespace sofa
