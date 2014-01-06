#ifndef COMPLIANT_CONSTANTASSEMBLEDSOLVER_H
#define COMPLIANT_CONSTANTASSEMBLEDSOLVER_H


#include "AssembledSolver.h"

namespace sofa {

namespace component {

namespace odesolver {
			


/** An AssembledSolver where the assembly is only perform at the first step
  @warning: only external forces can interact with a pre-assembled system
*/


class SOFA_Compliant_API ConstantAssembledSolver : public AssembledSolver {
  public:
				
    SOFA_CLASS(ConstantAssembledSolver, AssembledSolver);

  protected:

    /// a derivable function creating and calling the assembly visitor to create an AssembledSystem
    virtual void perform_assembly( const core::MechanicalParams *mparams, system_type& sys );
				

};

}
}
}



#endif
