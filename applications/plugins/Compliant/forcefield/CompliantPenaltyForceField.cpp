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

SOFA_DECL_CLASS(CompliantPenaltyForceField)
int CompliantPenaltyForceFieldClass = core::RegisterObject("Penalty ForceField")
    
#ifndef SOFA_FLOAT    
    .add< CompliantPenaltyForceField< Vec1dTypes > >(true)
#endif
#ifndef SOFA_DOUBLE
    .add< CompliantPenaltyForceField< Vec1fTypes > >()
#endif
    ;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API CompliantPenaltyForceField<Vec1dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API CompliantPenaltyForceField<Vec1fTypes>;
#endif

}
}
}
