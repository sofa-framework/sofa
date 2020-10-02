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
#ifndef SOFA_FLOAT
        .add< DiagonalStiffness< Vec1dTypes > >(true)
        .add< DiagonalStiffness< Vec3dTypes > >()
        .add< DiagonalStiffness< Vec6dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< DiagonalStiffness< Vec1fTypes > >()
        .add< DiagonalStiffness< Vec3fTypes > >()
        .add< DiagonalStiffness< Vec6fTypes > >()
#endif
        ;

SOFA_DECL_CLASS(DiagonalStiffness)

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API DiagonalStiffness<Vec1dTypes>;
template class SOFA_Compliant_API DiagonalStiffness<Vec3dTypes>;
template class SOFA_Compliant_API DiagonalStiffness<Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API DiagonalStiffness<Vec1fTypes>;
template class SOFA_Compliant_API DiagonalStiffness<Vec3fTypes>;
template class SOFA_Compliant_API DiagonalStiffness<Vec6fTypes>;
#endif

}
}
}
