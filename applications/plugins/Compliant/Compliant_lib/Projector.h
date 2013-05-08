#ifndef COMPLIANTDEV_PROJECTOR_H
#define COMPLIANTDEV_PROJECTOR_H

#include "initCompliant.h"
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa {
namespace component {
namespace linearsolver {


// TODO clarify whether we should make this a SOFA_DECL_CLASS with
// empty ::project method
struct SOFA_Compliant_API Projector : core::objectmodel::BaseObject {

	// SOFA_CLASS(Projector, sofa::core::objectmodel::BaseObject);
 
	virtual ~Projector(); 
	
	virtual void project(SReal* out, unsigned n) const = 0;
	
};

}
}
}


#endif
