#ifndef COMPLIANT_CONSTANTASSEMBLEDSOLVER_H
#define COMPLIANT_CONSTANTASSEMBLEDSOLVER_H


#include "CompliantImplicitSolver.h"

namespace sofa {

namespace component {

namespace odesolver {
			


/** An CompliantImplicitSolver where the assembly is only perform at the first step
  @warning: only external forces can interact with a pre-assembled system
*/


class SOFA_Compliant_API ConstantCompliantImplicitSolver : public CompliantImplicitSolver {
  public:
				
    SOFA_CLASS(ConstantCompliantImplicitSolver, CompliantImplicitSolver);

    virtual void reinit();

  protected:

    /// a derivable function creating and calling the assembly visitor to create an AssembledSystem
    virtual void perform_assembly( const core::MechanicalParams *mparams, system_type& sys );
				

};

}
}
}



#endif
