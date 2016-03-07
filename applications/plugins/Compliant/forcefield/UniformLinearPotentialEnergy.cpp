#include "UniformLinearPotentialEnergy.h"

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(UniformLinearPotentialEnergy)
int UniformLinearPotentialEnergyClass = core::RegisterObject("Linear potential energy")
    
#ifndef SOFA_FLOAT    
    .add< UniformLinearPotentialEnergy<Vec1dTypes> >(true)
#endif
#ifndef SOFA_DOUBLE
    .add< UniformLinearPotentialEnergy<Vec1fTypes> >()
#endif
        ;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API UniformLinearPotentialEnergy<Vec1dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API UniformLinearPotentialEnergy<Vec1fTypes>;
#endif

}
}
}
