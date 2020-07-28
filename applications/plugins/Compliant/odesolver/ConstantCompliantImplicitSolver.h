#ifndef COMPLIANT_CONSTANTASSEMBLEDSOLVER_H
#define COMPLIANT_CONSTANTASSEMBLEDSOLVER_H


#include "CompliantImplicitSolver.h"

namespace sofa {

namespace component {

namespace odesolver {
			


/** An CompliantImplicitSolver where the assembly is only performed at the first step
  @warning: only external forces can interact with a pre-assembled system

  @author Matthieu Nesme
  @date 2014
*/


class SOFA_Compliant_API ConstantCompliantImplicitSolver : public CompliantImplicitSolver {
  public:
				
    SOFA_CLASS(ConstantCompliantImplicitSolver, CompliantImplicitSolver);

    void reinit() override;

  protected:

    /// a derivable function creating and calling the assembly visitor to create an AssembledSystem
    void perform_assembly( const core::MechanicalParams *mparams, system_type& sys ) override;
				

};

}
}
}



#endif
