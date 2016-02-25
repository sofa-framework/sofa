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

SOFA_DECL_CLASS(PotentialEnergy)
int PotentialEnergyClass = core::RegisterObject("potential energy")
    
#ifndef SOFA_FLOAT    
    .add< PotentialEnergy< Vec1dTypes > >(true)
#endif
#ifndef SOFA_DOUBLE
    .add< PotentialEnergy< Vec1fTypes > >()
#endif
    ;


#ifndef SOFA_FLOAT
template class SOFA_Compliant_API PotentialEnergy<Vec1dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API PotentialEnergy<Vec1fTypes>;
#endif

}
}
}
