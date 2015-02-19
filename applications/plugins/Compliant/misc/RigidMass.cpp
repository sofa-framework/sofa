#include "RigidMass.h"
#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <Compliant/Compliant.h>

namespace sofa {

namespace component {

namespace mass {

SOFA_DECL_CLASS(RigidMass)

using namespace defaulttype;


// Register in the Factory
int RigidMassClass = core::RegisterObject("Mass for rigid bodies")


#ifndef SOFA_FLOAT
.add< RigidMass< Rigid3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< RigidMass< Rigid3fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API RigidMass<  Rigid3dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API RigidMass< Rigid3fTypes >;

#endif

}
}
}
