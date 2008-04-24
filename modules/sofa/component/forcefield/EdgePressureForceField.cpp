#include <sofa/component/forcefield/EdgePressureForceField.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>


#ifdef _WIN32
#include <windows.h>
#endif


namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

using std::cerr;
using std::cout;
using std::endl;

SOFA_DECL_CLASS(EdgePressureForceField)

int EdgePressureForceFieldClass = core::RegisterObject("EdgePressure")
#ifndef SOFA_FLOAT
        .add< EdgePressureForceField<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< EdgePressureForceField<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class EdgePressureForceField<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class EdgePressureForceField<Vec3fTypes>;
#endif


} // namespace forcefield

} // namespace component

} // namespace sofa
