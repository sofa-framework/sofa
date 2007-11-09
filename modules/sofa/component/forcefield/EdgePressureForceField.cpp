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

template class EdgePressureForceField<Vec3dTypes>;
template class EdgePressureForceField<Vec3fTypes>;

SOFA_DECL_CLASS(EdgePressureForceField)

int EdgePressureForceFieldClass = core::RegisterObject("EdgePressure")
        .add< EdgePressureForceField<Vec3dTypes> >()
        .add< EdgePressureForceField<Vec3fTypes> >()
        ;

} // namespace forcefield

} // namespace component

} // namespace sofa
