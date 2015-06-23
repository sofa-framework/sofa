#include "CompliantStaticSolver.h"

#include <sofa/core/ObjectFactory.h>


namespace sofa {
namespace component {
namespace odesolver {

CompliantStaticSolver::CompliantStaticSolver()
    : epsilon(initData(&epsilon, (SReal)1e-14, "epsilon", "division by zero threshold")),
      line_search(initData(&line_search, true, "line_search", "perform line search")),
      conjugate(initData(&conjugate, true, "conjugate", "conjugate descent directions")),
      ls_precision(initData(&ls_precision, (SReal)1e-7, "ls_precision", "line search precision")),
      ls_iterations(initData(&ls_iterations, unsigned(10), "ls_iterations", "line search iterations"))
{
    
}


struct CompliantStaticSolver::helper {

    simulation::common::VectorOperations vec;
    simulation::common::MechanicalOperations mec;

    core::behavior::MultiVecDeriv dx;
    core::behavior::MultiVecDeriv f;
    
    helper(const core::ExecParams* params,
           core::objectmodel::BaseContext* ctx) : vec(params, ctx),
                                                  mec(params, ctx),
                                                  dx( &vec, core::VecDerivId::dx() ),
                                                  f( &vec, core::VecDerivId::force() )    {
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

    template<class A, class B>
    void set(const A& res,
             const A& a,
             const B& b,
             SReal lambda) {
        vec.v_op(res, a, b, lambda);
    }


    
};

CompliantStaticSolver::ls_info::ls_info()
    : eps(0),
      iterations(15),
      precision(1e-7),
      step(1e-5){
    
}

void CompliantStaticSolver::secant_ls(helper& op,
                                      core::MultiVecCoordId pos,
                                      core::MultiVecDerivId dir,
                                      const ls_info& info) {
    // assumes op.f contains forces already !
    SReal dg = 0;
    SReal dx = 0;

    SReal g_old = 0;
    SReal total = 0;
    
    for(unsigned k = 0, n = info.iterations; k < n; ++k) {

        const SReal g = op.dot(op.f, dir);

        // std::cout << "line search (secant) " << k << " " << g << std::endl;
        
        if( std::abs(g) <= info.precision ) break;
        
        dg = g - g_old;
        
        const SReal dx_prev = dx;

        // fallback on fixed step
        dx = info.step;

        // secant method
        if( (k > 0) && (std::abs(dg) > info.eps)) {
            dx = -(dx_prev / dg) * g;
        }
        
        total += dx;
        
        // move dx along dir
        op.set(pos, pos, dir, dx);
        op.mec.propagateX(pos, true);

        // update forces
        op.forces( op.f );

        g_old = g;
    }
    
    
}



SOFA_DECL_CLASS(CompliantStaticSolver)
static int CompliantStaticSolverClass = core::RegisterObject("Static solver")
    .add< CompliantStaticSolver >();


    void CompliantStaticSolver::solve(const core::ExecParams* params,
                                SReal dt,
                                core::MultiVecCoordId pos,
                                core::MultiVecDerivId vel) {

        helper op(params, getContext() );
        
        op.mec.mparams.setImplicit(false);
        op.mec.mparams.setEnergy(false);
        
        // (negative) gradient
        
        // descent direction
        op.realloc(descent);

        // obtain (projected) gradient
        op.forces( op.f );
        
        // note: we *could* skip the above when line-search is on
        // after the first iteration, but we would miss any change in
        // the scene (e.g. mouse interaction)
        
        const SReal current = op.dot(op.f, op.f);
        SReal beta = 0;

        const SReal eps = epsilon.getValue();
        if( conjugate.getValue() && std::abs(previous) > eps ) {

            // polak ribiere
            {
                if(iteration > 0) {
                    beta = (current - op.dot(vel, op.f) ) / previous;

                    // direction reset
                    beta = std::max((SReal)0.0, beta);
                }
            }
            
        }

        // conjugation
        op.vec.v_op(descent, op.f, descent, beta);

        // polak-ribiere
        {
            // backup previous f to vel
            op.vec.v_eq(vel, op.f);
        }

        
        // line search
        if( line_search.getValue() ) {
            ls_info info;

            info.eps = eps;
            info.precision = ls_precision.getValue();
            info.iterations = ls_iterations.getValue();
            info.step = dt;
            
            secant_ls(op, pos, descent.id(), info);
        } else {
            // fixed step
            op.set(pos, pos, descent.id(), dt);
        }
        
        // next iteration
        previous = current;

        
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
