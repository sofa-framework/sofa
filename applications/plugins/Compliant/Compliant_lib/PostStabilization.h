#ifndef COMPLIANTDEV_POSTSTABILIZATION_H
#define COMPLIANTDEV_POSTSTABILIZATION_H

#include "initCompliant.h"

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa {
namespace component {
namespace odesolver {

class SOFA_Compliant_API PostStabilization : public core::objectmodel::BaseObject {
  public:

	SOFA_CLASS(PostStabilization, core::objectmodel::BaseObject);
};

}
}
}



#endif
