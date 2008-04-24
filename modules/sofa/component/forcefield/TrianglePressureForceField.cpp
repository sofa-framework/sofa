#include <sofa/component/forcefield/TrianglePressureForceField.inl>
#include <sofa/core/ObjectFactory.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <vector>
#include <set>
#include <sofa/defaulttype/Vec3Types.h>

#ifdef _WIN32
#include <windows.h>
#endif

// #define DEBUG_TRIANGLEFEM

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


using namespace sofa::defaulttype;

SOFA_DECL_CLASS(TrianglePressureForceField)

int TrianglePressureForceFieldClass = core::RegisterObject("TrianglePressure")
#ifndef SOFA_FLOAT
        .add< TrianglePressureForceField<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TrianglePressureForceField<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class TrianglePressureForceField<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class TrianglePressureForceField<Vec3fTypes>;
#endif


} // namespace forcefield

} // namespace component

} // namespace sofa
