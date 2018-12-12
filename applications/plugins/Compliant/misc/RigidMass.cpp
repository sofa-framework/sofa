#include "RigidMass.h"
#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <Compliant/config.h>

namespace sofa {

namespace component {

namespace mass {

using namespace defaulttype;


// Register in the Factory
int RigidMassClass = core::RegisterObject("Mass for rigid bodies")


.add< RigidMass< Rigid3Types > >()

;

template class SOFA_Compliant_API RigidMass<  Rigid3Types >;


}
}
}
