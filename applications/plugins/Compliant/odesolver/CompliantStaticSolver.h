#ifndef COMPLIANT_STATICSOLVER_H
#define COMPLIANT_STATICSOLVER_H


#include <Compliant/config.h>

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/MultiVec.h>

#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>

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

    struct helper;
    struct potential_energy;

    struct ls_info {
        ls_info();

        SReal eps;              // zero threshold for divisions
        unsigned iterations;    // iteration count
        SReal precision;        // stop criterion
        SReal fixed_step;       // fallback fixed step
        SReal bracket_step;        // init bracketing step
    };
    
    static void ls_secant(helper& op,
                          core::MultiVecCoordId pos,
                          core::MultiVecDerivId dir,
                          const ls_info& info);

    static void ls_brent(helper& op,
                         core::MultiVecCoordId pos,
                         core::MultiVecDerivId dir,
                         const ls_info& info,
                         core::MultiVecCoordId tmp);
    

    // descent direction
    core::behavior::MultiVecDeriv dir, lambda;

    // temporary position
    core::behavior::MultiVecCoord tmp;
    
    SReal previous;
    unsigned iteration;

    Data<SReal> epsilon; ///< division by zero threshold

    Data<unsigned> line_search;
    enum {
        LS_NONE = 0,
        LS_BRENT,
        LS_SECANT
    };

    Data<bool> conjugate; ///< conjugate descent directions
    
    Data<SReal> ls_precision; ///< line search precision
    Data<unsigned> ls_iterations;
    Data<SReal> ls_step;

    SReal augmented;
    
};

}
}
}



#endif
