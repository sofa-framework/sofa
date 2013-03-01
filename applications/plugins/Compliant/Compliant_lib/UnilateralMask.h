#ifndef UNILATERALMASK_H
#define UNILATERALMASK_H

#include "initCompliant.h"
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa {
namespace component {
namespace linearsolver {

class UnilateralMask  : public virtual core::objectmodel::BaseObject {
	typedef helper::vector<SReal > mask_type;

	// 1 or -1  is positive (geq) or negative (leq)
	Data< mask_type > mask;

	// if mask is not set in xml, use this value to initialize it
	Data< SReal > value;
public:
	SOFA_CLASS(UnilateralMask,sofa::core::objectmodel::BaseObject);

	UnilateralMask();

	void init();

	// writes mask into out buffer, returns size
	unsigned write(SReal* out) const;
	
};

}
}
}

#endif
