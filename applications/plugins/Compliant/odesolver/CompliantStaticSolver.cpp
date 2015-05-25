#include "CompliantStaticSolver.h"

#include <sofa/core/ObjectFactory.h>


namespace sofa {
namespace component {
namespace odesolver {

CompliantStaticSolver::CompliantStaticSolver()
    : epsilon(initData(&epsilon, 1e-5, "epsilon", "division by zero threshold")),
      beta_max(initData(&beta_max, 10.0, "beta_max", "magic")),
      line_search(initData(&line_search, true, "line_search", "line search")),
      fletcher_reeves(initData(&fletcher_reeves, true, "fletcher_reeves", "fletcher-reeves")) {

}


struct helper {
    sofa::simulation::common::VectorOperations vec;
    sofa::simulation::common::MechanicalOperations mec;
    core::behavior::MultiVecDeriv dx;
    
    helper(const core::ExecParams* params,
           core::objectmodel::BaseContext* ctx) : vec(params, ctx),
                                                  mec(params, ctx),
                                                  dx( &vec, core::VecDerivId::dx() ){
        
    }

    // TODO const args ?
    SReal dot(core::MultiVecDerivId lhs, core::MultiVecDerivId rhs)  {
        vec.v_dot(lhs, rhs);
        return vec.finish();
    }

    // arg can't be dx !
    void K(core::MultiVecDerivId res, core::MultiVecDerivId arg) {
        vec.v_eq(dx, arg);
        vec.v_clear( res );
        
        mec.addMBKdx( res , 0, 0, -1, true, true);
        mec.projectResponse( res );
    }

    void forces(core::MultiVecDerivId res) {
        mec.computeForce( res );
        mec.projectResponse( res );
    }

    void realloc(core::behavior::MultiVecDeriv& res) {
        res.realloc( &vec, false, true );
    }
           
};



SOFA_DECL_CLASS(CompliantStaticSolver)
static int CompliantStaticSolverClass = core::RegisterObject("Static solver")
    .add< CompliantStaticSolver >();


    void CompliantStaticSolver::solve(const core::ExecParams* params,
                                SReal dt,
                                core::MultiVecCoordId pos,
                                core::MultiVecDerivId vel) {

        helper op(params, getContext() );
        
        op.mec.mparams.setImplicit(true);
        op.mec.mparams.setEnergy(true);
        
        // (negative) gradient
        core::behavior::MultiVecDeriv f( &op.vec, core::VecDerivId::force() );
        
        // descent direction
        op.realloc(descent);

        // obtain (projected) gradient
        op.forces( f );
        
        SReal f2 = op.dot(f, f);
        SReal beta = 0;

        const SReal eps = epsilon.getValue();
        
        if( fletcher_reeves.getValue() && std::abs(previous) > 0 ) {

            // fletcher-reeves
            beta = f2 / previous;

        }

        // std::cout << "g2, " << f2 << ", beta " << beta << std::endl;
        //beta = 0;
        
        // reset in case we increased gradient
        if( beta > beta_max.getValue() ) {
            std::cout << iteration << ", reset" << std::endl;
            std::cout << "beta: " << beta
                      << ", g2: " << previous
                      << ", g2_next: " << f2 << std::endl;
            std::cout << "beta max " << beta_max.getValue() << std::endl;

            beta = 0;
            f2 = 0;
        }

        // conjugation
        op.vec.v_op(descent, f, descent, beta);

        
        // TODO an actual line-search
        SReal alpha = dt;

        // quadratic line-search
        if( line_search.getValue() ) {

            // vel = K * descent
            op.K(vel, descent);

            const SReal num = op.dot(f, descent);
            const SReal den = op.dot(vel, descent);
            
            std::cout << num << " / " << den << " = " << num / den << std::endl;

            if( den <= eps ) {

                // descent direction is zero/negative curvature
                std::cout << iteration << ", derp" << std::endl;
                f2 = 0;
                
            } else {
                alpha = num / den;
            }
            
        }
        
        

        // position integration
        op.vec.v_op(pos, pos, descent, alpha);
        
        // next iteration
        previous = f2;
        ++iteration;
    }




    void CompliantStaticSolver::reset() {
        iteration = 0;
       
    }

    void CompliantStaticSolver::init() {
        previous = 0;
    }







}
}
}
