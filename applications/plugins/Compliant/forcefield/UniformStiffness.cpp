#include "UniformStiffness.inl"
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

using namespace sofa::defaulttype;

// Register in the Factory
int UniformStiffnessClass = core::RegisterObject("Uniform stiffness")
#ifndef SOFA_FLOAT
        .add< UniformStiffness< Vec1dTypes > >(true)
        .add< UniformStiffness< Vec2dTypes > >()
        .add< UniformStiffness< Vec3dTypes > >()
        .add< UniformStiffness< Vec6dTypes > >()
#endif
#ifndef SOFA_DOUBLE
#ifdef SOFA_FLOAT
        .add< UniformStiffness< Vec1fTypes > >(true)
#else
        .add< UniformStiffness< Vec1fTypes > >()
#endif
        .add< UniformStiffness< Vec2fTypes > >()
        .add< UniformStiffness< Vec3fTypes > >()
        .add< UniformStiffness< Vec6fTypes > >()
#endif
        ;

SOFA_DECL_CLASS(UniformStiffness)

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API UniformStiffness<Vec1dTypes>;
template class SOFA_Compliant_API UniformStiffness<Vec2dTypes>;
template class SOFA_Compliant_API UniformStiffness<Vec3dTypes>;
template class SOFA_Compliant_API UniformStiffness<Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API UniformStiffness<Vec1fTypes>;
template class SOFA_Compliant_API UniformStiffness<Vec2fTypes>;
template class SOFA_Compliant_API UniformStiffness<Vec3fTypes>;
template class SOFA_Compliant_API UniformStiffness<Vec6fTypes>;
#endif

}
}
}
