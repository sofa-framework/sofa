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

int UniformLinearPotentialEnergyClass = core::RegisterObject("Linear potential energy")
    
    .add< UniformLinearPotentialEnergy<Vec1Types> >(true)

        ;


template class SOFA_Compliant_API UniformLinearPotentialEnergy<Vec1Types>;


}
}
}
