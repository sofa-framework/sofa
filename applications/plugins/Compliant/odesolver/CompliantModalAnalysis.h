#ifndef COMPLIANT_CompliantModalAnalysis_H
#define COMPLIANT_CompliantModalAnalysis_H


#include "CompliantImplicitSolver.h"




namespace sofa {


namespace component {



namespace odesolver {
			



class SOFA_Compliant_API CompliantModalAnalysis : public CompliantImplicitSolver {

    public:



				
    SOFA_CLASS(CompliantModalAnalysis, sofa::component::odesolver::CompliantImplicitSolver);


    typedef linearsolver::AssembledSystem system_type;
				
    // OdeSolver API
    virtual void solve(const core::ExecParams* params,
                       SReal dt,
                       core::MultiVecCoordId posId,
                       core::MultiVecDerivId velId);


    CompliantModalAnalysis();
    virtual ~CompliantModalAnalysis();


};

}
}
}



#endif
