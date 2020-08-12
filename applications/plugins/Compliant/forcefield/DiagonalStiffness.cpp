#include "DiagonalStiffness.inl"
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
int DiagonalStiffnessClass = core::RegisterObject("Diagonal stiffness")
        .add< DiagonalStiffness< Vec1Types > >(true)
        .add< DiagonalStiffness< Vec3Types > >()
        .add< DiagonalStiffness< Vec6Types > >()

        ;

template class SOFA_Compliant_API DiagonalStiffness<Vec1Types>;
template class SOFA_Compliant_API DiagonalStiffness<Vec3Types>;
template class SOFA_Compliant_API DiagonalStiffness<Vec6Types>;


}
}
}
