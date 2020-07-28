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
        .add< UniformStiffness< Vec1Types > >(true)
        .add< UniformStiffness< Vec2Types > >()
        .add< UniformStiffness< Vec3Types > >()
        .add< UniformStiffness< Vec6Types > >()

        ;

template class SOFA_Compliant_API UniformStiffness<Vec1Types>;
template class SOFA_Compliant_API UniformStiffness<Vec2Types>;
template class SOFA_Compliant_API UniformStiffness<Vec3Types>;
template class SOFA_Compliant_API UniformStiffness<Vec6Types>;


}
}
}
