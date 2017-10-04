#include "CompliantStaticSolver.h"

#include <sofa/core/ObjectFactory.h>

#include <boost/math/tools/minima.hpp>
#include <tuple>

#include "../utils/nr.h"
#include "../utils/scoped.h"

#include "../constraint/Constraint.h"

namespace sofa {
namespace component {
namespace odesolver {

// TODO use std::numeric_limits

CompliantStaticSolver::CompliantStaticSolver()
    : epsilon(initData(&epsilon, (SReal)1e-16, "epsilon", "division by zero threshold")),
      line_search(initData(&line_search, unsigned(LS_SECANT), "line_search",
                           "line search method, 0: none (use dt), 1: brent, 2: secant (default). (warning: brent does not work with constraints.")),
      conjugate(initData(&conjugate, true, "conjugate", "conjugate descent directions")),
      ls_precision(initData(&ls_precision, (SReal)1e-8, "ls_precision", "line search precision")),
      ls_iterations(initData(&ls_iterations, unsigned(10), "ls_iterations",
                             "line search iterations")),
      ls_step(initData(&ls_step, (SReal)1e-8, "ls_step",
                       "line search bracketing step (should not cross any extrema from current position"))
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

    template<class Vec>
    void realloc(Vec& res) {
        res.realloc( &vec, false, true );
    }

    
    template<class Res, class A, class B>
    void set(Res& res,
             A& a,
             B& b,
             SReal lambda) {
        vec.v_op(res, a, b, lambda);
    }

    template<class A, class B>
    void set(A& res,
             B& b) {
        vec.v_eq(res, b);
    }
    
};






CompliantStaticSolver::ls_info::ls_info()
    : eps(0),
      iterations(15),
      precision(1e-7),
      fixed_step(1e-5),
      bracket_step(1e-8) {
    
}


// minimizes (implicit) potential function in direction dir) by
// zeroing g = grad^T dir using secant method

// precond:
// - op.f contains forces for current pos
void CompliantStaticSolver::ls_secant(helper& op,
                                      core::MultiVecCoordId pos,
                                      core::MultiVecDerivId dir,
                                      const ls_info& info) {
    scoped::timer step("ls_secant");
    
    SReal dg = 0;
    SReal dx = 0;

    SReal g_prev = 0;
    SReal total = 0;

    // slightly damp newton iterations TODO hardcode
    static const SReal damping = 1e-14;

    SReal fixed = info.fixed_step;
    
    for(unsigned k = 0, n = info.iterations; k < n; ++k) {

        if( k ) {
            // update forces
            op.mec.projectPosition(pos); // apply projective constraints
            op.mec.propagateX(pos);
            op.forces( op.f );
        }
        
        const SReal g = op.dot(op.f, dir);

        // std::cout << "line search (secant) " << k << " "  << total << " " << g << std::endl;

        // are we done ?
        if( std::abs(g) <= info.precision ) break;
        
        dg = g - g_prev;
        
        const SReal dx_prev = dx;

        // fallback on fixed step
        dx = fixed;

        // (damped) secant method
        if( k && (std::abs(dg) > info.eps)) {
            dx = -(dx_prev / (dg + damping)) * g;
        } else {
            
            // try to move more to change function
            fixed *= 2;
            
        }
        
        total += dx;
        
        // move dx along dir
        op.set(pos, pos, dir, dx);
            
        // next
        g_prev = g;
    }
    
}



struct CompliantStaticSolver::potential_energy {

    // WARNING: backup/restore pos on scope entry/exit
    potential_energy(helper& op,
                     const core::MultiVecCoordId& pos,
                     const core::MultiVecDerivId& dir,
                     const core::MultiVecCoordId& tmp,
                     bool restore = true)
        : op(op),
          pos(pos),
          dir(dir),
          tmp(tmp),
          restore(restore) {

        // backup start position
        op.set(tmp, pos);
    }
    
    
    helper& op;

    const core::MultiVecCoordId& pos;
    const core::MultiVecDerivId& dir;
    const core::MultiVecCoordId& tmp;    

    // do we restore pos in dtor ?
    bool restore;
    
    SReal operator()(SReal x) const {

        // move to x along dir
        op.set(pos, tmp, dir, x);
        
        // update forces/energy
        op.mec.projectPosition(pos); // apply projective constraints
        op.mec.propagateX(pos);

        // TODO apparently, this is not needed
        // op.forces(op.f);

        // note: potential energy only works for pos !
        SReal dummy, result;
        op.mec.computeEnergy(dummy, result);

        // std::cout << "potential energy: " << x << ", " << result << std::endl;
        
        return result;
    }

    
    ~potential_energy() {
        if( restore ) {
            op.set(pos, tmp);
        }
    }
    
};

void CompliantStaticSolver::ls_brent(helper& op,
                                     core::MultiVecCoordId pos,
                                     core::MultiVecDerivId dir,
                                     const ls_info& info,
                                     core::MultiVecCoordId tmp) {
    scoped::timer step("ls_brent");
    typedef utils::nr::optimization<SReal> opt;
    
    opt::func_call a, b, c;
    
    a.x = 0;
    b.x = info.bracket_step;

    opt::func_call res;

    {
        // we need the scope because f might reset pos on scope exit
        const potential_energy f(op, pos, dir, tmp, false);

        opt::minimum_bracket(a, b, c, f);

        // std::cout << "bracketing: " << a.x << ", " << c.x << std::endl;
    
        // TODO compute this from precision
        const int bits = 32;
        {
            boost::uintmax_t iter = info.iterations;
            std::tie(res.x, res.f) = boost::math::tools::brent_find_minima(f,
                                                               a.x, c.x,
                                                               bits,
                                                               iter);
            // std::cout << "brent: " << res.x << std::endl;
        }

    }
    
    // TODO make sure the last function call is the closest to the optimium
    // op.set(pos, pos, dir, res.x);
    
    // TODO do we want to do this ?
    // op.mec.projectPosition(pos); // apply projective constraints
    // op.mec.propagateX(pos);
    // op.forces(op.f);
    
}


// accumulate force in an external vector
class AugmentedLagrangianVisitor : public simulation::MechanicalVisitor {

public:
    AugmentedLagrangianVisitor(const sofa::core::MechanicalParams* mparams,
                               core::MultiVecId id)
        : MechanicalVisitor(mparams),
          id(id) { }

    core::MultiVecId id;
    
    Result mstate(simulation::Node* node,
                  core::behavior::BaseMechanicalState* mm) {

        linearsolver::Constraint* c = node->get<linearsolver::Constraint>(core::objectmodel::BaseContext::Local);


        if( c ) {
            // TODO project ?
            
            // add force to external force
            mm->vOp(params, id.getId(mm), core::ConstVecId::force() );            
        }

        return RESULT_CONTINUE;
    }
    
    virtual Result fwdMappedMechanicalState(simulation::Node* node,
                                            core::behavior::BaseMechanicalState* mm) {
        return mstate(node, mm);
    }

    virtual Result fwdMechanicalState(simulation::Node* node,
                                      core::behavior::BaseMechanicalState* mm) {
        return mstate(node, mm);
    }

};


// somehow setting external force directly does not work 
class WriteExternalForceVisitor : public simulation::MechanicalVisitor {
public:
    WriteExternalForceVisitor(const sofa::core::MechanicalParams* mparams,
                              core::MultiVecId id)
        : MechanicalVisitor(mparams),
          id(id) { }


    core::MultiVecId id;
    
    Result mstate(simulation::Node* /*node*/,
                  core::behavior::BaseMechanicalState* mm) {

        // add force to external force

        // we need to add to externalForce (gravity)
        mm->vOp(params, core::VecId::externalForce(), id.getId(mm));
        // core::VecId::externalForce(),
        // , 1.0 );
        
        return RESULT_CONTINUE;
    }
    
    virtual Result fwdMappedMechanicalState(simulation::Node* node,
                                            core::behavior::BaseMechanicalState* mm) {
        return mstate(node, mm);
    }

    virtual Result fwdMechanicalState(simulation::Node* node,
                                      core::behavior::BaseMechanicalState* mm) {
        return mstate(node, mm);
    }

};


SOFA_DECL_CLASS(CompliantStaticSolver)
int CompliantStaticSolverClass = core::RegisterObject("Static solver")
    .add< CompliantStaticSolver >();


    void CompliantStaticSolver::solve(const core::ExecParams* params,
                                SReal dt,
                                core::MultiVecCoordId pos,
                                core::MultiVecDerivId vel) {

        helper op(params, getContext() );

        // mparams setup
        op.mec.mparams.setImplicit(false);
        op.mec.mparams.setEnergy(true);
        
        // descent direction
        op.realloc(dir);

        // lagrange multipliers
        op.realloc(lambda);

        // some work vector for postions
        op.realloc(tmp);

        // first iteration
        if(!iteration) {
            op.vec.v_clear( lambda );
            previous = 0;
            
        }
        
        // why on earth does this dot work ?!

        // core::behavior::MultiVecDeriv ext(&op.vec, core::VecDerivId::externalForce() );
        // op.set(ext, lambda);
        {
            WriteExternalForceVisitor vis(&op.mec.mparams, lambda.id());
            getContext()->executeVisitor(&vis, true);

            // core::behavior::MultiVecDeriv ext(&op.vec, core::VecDerivId::externalForce() );
            // std::cout << ext << std::endl;
        }
        
        // obtain (projected) gradient
        op.forces( op.f );
        
        // note: we *could* skip the above when line-search is on
        // after the first iteration, but we would miss any change in
        // the scene (e.g. mouse interaction)

        // polar-ribiere
        const SReal current = op.dot(op.f, op.f);

        if(!iteration) {
            // something large at first ?
            
            // TODO figure out a reasonable default
            augmented = std::sqrt(current) / 2.0;
        }


        
        SReal beta = 0;

        const SReal eps = epsilon.getValue();
        if( conjugate.getValue() && std::abs(previous) > eps ) {

            {
                if(iteration > 0) {
                    // polak-ribiere
                    beta = (current - op.dot(vel, op.f) ) / previous;

                    // dai-yuan
                    // beta = current / (op.dot(dir, vel) - op.dot(dir, op.f) );
                }

                // direction reset
                // beta = std::max(0.0, beta);
            }
            
        }

        // conjugation
        op.set(dir, op.f, dir, beta);

        // polak-ribiere
        {
            // backup previous f to vel
            op.set(vel, op.f);
        }

        
        // line search
        const unsigned ls = line_search.getValue();
        if( ls ) {
            ls_info info;

            info.eps = eps;
            info.precision = ls_precision.getValue();
            info.iterations = ls_iterations.getValue();
            info.fixed_step = dt; 
            info.bracket_step = ls_step.getValue();
            
            switch( ls ) {
                
            case LS_SECANT:
                ls_secant(op, pos, dir.id(), info);
                break;
                
            case LS_BRENT:
                ls_brent(op, pos, dir.id(), info, tmp.id());
                break;
                
            default:
                throw std::runtime_error("bad line-search");
            }
            
        } else {
            // fixed step
            op.set(pos, pos, dir.id(), dt);
        }

        const SReal error = std::sqrt( op.dot(op.f, op.f) );

        if( f_printLog.getValue() ) {
            sout << "forces norm: " << error << sendl;
        }
        
        // augmented lagrangian
        if( error <= augmented ) {
            
            // TODO don't waste time if we have no constraints
            op.mec.projectPosition(pos); // apply projective constraints
            op.mec.propagateX(pos);
            op.forces(op.f);

            AugmentedLagrangianVisitor vis(&op.mec.mparams, lambda.id() );
            getContext()->executeVisitor( &vis, true );

            // TODO should we reset CG ?
            // op.vec.v_clear(dir);
            
            augmented /= 2;

            if( f_printLog.getValue() ) {
                sout << "augmented lagrangian threshold: " << augmented << sendl;
            }
                    
        }


        // next iteration
        previous = current;

        ++iteration;
    }




    void CompliantStaticSolver::reset() {
        iteration = 0;

    }

    void CompliantStaticSolver::init() {
        
    }







}
}
}
