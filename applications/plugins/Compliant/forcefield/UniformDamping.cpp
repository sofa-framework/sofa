#include "UniformDamping.h"
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace forcefield {

using namespace sofa::defaulttype;

static const int handle = core::RegisterObject("Uniform damping")
#ifndef SOFA_FLOAT
    .add< UniformDamping< Vec1dTypes > >()
    .add< UniformDamping< Vec2dTypes > >()
    .add< UniformDamping< Vec3dTypes > >()
    .add< UniformDamping< Vec6dTypes > >()
    .add< UniformDamping< Rigid3dTypes > >()    
#endif
#ifndef SOFA_DOUBLE
#ifdef SOFA_FLOAT
    .add< UniformDamping< Vec1fTypes > >()
#else
    .add< UniformDamping< Vec1fTypes > >()
#endif
    .add< UniformDamping< Vec2fTypes > >()
    .add< UniformDamping< Vec3fTypes > >()
    .add< UniformDamping< Vec6fTypes > >()
    .add< UniformDamping< Rigid3fTypes > >()        
#endif
    ;

SOFA_DECL_CLASS(UniformDamping);

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API UniformDamping<Vec1dTypes>;
template class SOFA_Compliant_API UniformDamping<Vec2dTypes>;
template class SOFA_Compliant_API UniformDamping<Vec3dTypes>;
template class SOFA_Compliant_API UniformDamping<Vec6dTypes>;
template class SOFA_Compliant_API UniformDamping<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API UniformDamping<Vec1fTypes>;
template class SOFA_Compliant_API UniformDamping<Vec2fTypes>;
template class SOFA_Compliant_API UniformDamping<Vec3fTypes>;
template class SOFA_Compliant_API UniformDamping<Vec6fTypes>;
template class SOFA_Compliant_API UniformDamping<Rigid3fTypes>;
#endif

}
}
}
