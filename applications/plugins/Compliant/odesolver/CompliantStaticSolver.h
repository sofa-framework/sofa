#ifndef COMPLIANT_STATICSOLVER_H
#define COMPLIANT_STATICSOLVER_H


#include "initCompliant.h"

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/MultiVec.h>

#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/MechanicalOperations.h>
#include <sofa/simulation/common/VectorOperations.h>

#include <sofa/helper/OptionsGroup.h>

namespace sofa {

namespace simulation {

namespace common {
class MechanicalOperations;
class VectorOperations;
}

}

namespace component {

namespace linearsolver {

}


namespace odesolver {

class SOFA_Compliant_API CompliantStaticSolver : public sofa::core::behavior::OdeSolver {

  public:

				
	SOFA_CLASS(CompliantStaticSolver, sofa::core::behavior::OdeSolver);

    virtual void init();

    // OdeSolver API
    virtual void solve(const core::ExecParams* params,
                       SReal dt,
                       core::MultiVecCoordId posId,
                       core::MultiVecDerivId velId);

    virtual void reset();


    CompliantStaticSolver();
    
  protected:


    // descent direction
    core::behavior::MultiVecDeriv descent;


    SReal previous;
    unsigned iteration;

    Data<SReal> epsilon, beta_max;


    Data<bool> line_search, fletcher_reeves;
};

}
}
}



#endif
