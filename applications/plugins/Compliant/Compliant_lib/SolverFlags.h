#ifndef COMPLIANT_SOLVER_FLAGS_H
#define COMPLIANT_SOLVER_FLAGS_H

#include "initCompliant.h"
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa {
namespace component {
namespace linearsolver {


// TODO subclass this as user-friendly classes (UnilateralFlags ... )
class SOFA_Compliant_API SolverFlags  : public virtual core::objectmodel::BaseObject {
  public:
	typedef unsigned value_type;
	typedef helper::vector< value_type > flags_type;
	
  public:
	
	// per-dof flags, usually hanging around compliant dofs
	Data< flags_type > flags;
	
	// if flags are not set in xml, use this value to initialize it
	Data< value_type > value;
	
	// TODO using boost::any here would probably be safer but who cares
	// :)
	typedef void* data_type;
	data_type data;
	
  public:
	
	// standard flags
	enum {
		NO_FLAG = 0x0,
		UNILATERAL_FLAG = 0x1,			// unilateral constraint
		NEGATIVE_FLAG = 0x10, 			// negative unilateral constraint
		FRICTION_FLAG = 0x100,
		USER_FLAG = 0x10000000				// start of user-defined flags
	};
	
	SOFA_CLASS(SolverFlags, sofa::core::objectmodel::BaseObject);

	SolverFlags();
	
	void init();
	
	// writes mask into out buffer, returns written count
	unsigned write(value_type* out) const;
	

};

}
}
}

#endif
