#include "PotentialEnergy.h"

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

using namespace sofa::defaulttype;

int PotentialEnergyClass = core::RegisterObject("potential energy")
    
    .add< PotentialEnergy< Vec1Types > >(true)

    ;


template class SOFA_Compliant_API PotentialEnergy<Vec1Types>;


}
}
}
