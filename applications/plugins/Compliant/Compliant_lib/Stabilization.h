#ifndef COMPLIANTDEV_STABILIZATION_H
#define COMPLIANTDEV_STABILIZATION_H

#include "initCompliant.h"

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa {
namespace component {
namespace odesolver {

class SOFA_Compliant_API Stabilization : public core::objectmodel::BaseObject {
  public:

	SOFA_CLASS(Stabilization, core::objectmodel::BaseObject);

	// TODO add fancy options here (pre/post/multistep/...)

	
};

}
}
}



#endif
