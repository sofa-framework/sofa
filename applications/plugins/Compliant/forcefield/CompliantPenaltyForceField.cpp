#include "CompliantPenaltyForceField.h"

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

using namespace sofa::defaulttype;

int CompliantPenaltyForceFieldClass = core::RegisterObject("Penalty ForceField")
    
    .add< CompliantPenaltyForceField< Vec1Types > >(true)

    ;


template class SOFA_Compliant_API CompliantPenaltyForceField<Vec1Types>;


}
}
}
