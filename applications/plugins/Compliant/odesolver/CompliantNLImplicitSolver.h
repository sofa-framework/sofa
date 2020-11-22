#ifndef COMPLIANT_CompliantNLImplicitSolver_H
#define COMPLIANT_CompliantNLImplicitSolver_H

#include <Compliant/config.h>
#include <Compliant/odesolver/CompliantImplicitSolver.h>
#include <sofa/simulation/MechanicalOperations.h>


namespace sofa {
namespace component {
namespace odesolver {


/** Implicit solver with Newton iterations to solve the non-linear implicit equation at arbitrary precision.

    The method computes the next velocity \f$ v(t+h) \f$, such that equation \f$ M (v(t+h)-v(t)) - hf \f$ is satisfied, where the usual conventions of this integrator are (see parent class):
    - \f$ f = \alpha f(t+h) + (1-\alpha) f(t) \f$ is a weighted sum of the current and next force,
    - \f$ x(t+h) = x(t) + h (\beta v(t+h) + (1-\beta) v(t)\f$ is the next position, where the next force is computed, using the next velocity.

    The method iteratively improves an approximate solution by solving a linear equation system based on the Jacobian of the residual of the equation to satisfy.


    TODO: more doc about constraints

  @author François Faure & Matthieu Nesme, 2013

*/


class SOFA_Compliant_API CompliantNLImplicitSolver : public CompliantImplicitSolver {
  public:
				
	SOFA_CLASS(CompliantNLImplicitSolver, CompliantImplicitSolver);
				
    Data<SReal> precision;        ///< Convergence threshold of the Newton method (L-infinite norm of the residual)
    Data<bool> relative;          ///< Relative precision?
    Data<unsigned> iterations;    ///< Max number of iterations of the Newton method.
    Data<SReal> newtonStepLength; ///< the portion of correction applied while it is converging (can be applied several times - until complete correction - at each Newton iteration)
//    Data<bool> staticSolver;      ///< solve a static analysis (dynamics otherwise) WIP

	CompliantNLImplicitSolver();
    ~CompliantNLImplicitSolver() override{}

    void cleanup() override;
	
    // OdeSolver API
    void solve(const core::ExecParams* params,
                       SReal dt,
                       core::MultiVecCoordId posId,
                       core::MultiVecDerivId velId) override;

    typedef Eigen::Map<vec> chuck_type;

  protected:



    using CompliantImplicitSolver::integrate;
    /// newPos = oldPos + beta*h*vel
    void integrate( SolverOperations& sop, core::MultiVecCoordId oldPos, core::MultiVecCoordId newPos, core::MultiVecDerivId vel );

    /// compute a first approximation with the regular, linearized system
    void firstGuess( SolverOperations& sop, core::MultiVecCoordId posId, core::MultiVecDerivId velId );

    void compute_forces(SolverOperations& sop, core::behavior::MultiVecDeriv& f, core::behavior::MultiVecDeriv* f_k=NULL ) override;

    /// Residual of the non-linear implicit integration equation
    SReal compute_residual( SolverOperations sop, core::MultiVecDerivId residual, core::MultiVecCoordId newX, const core::MultiVecDerivId newV, core::MultiVecDerivId newF, core::MultiVecCoordId oldX, core::MultiVecDerivId oldV, core::MultiVecDerivId oldF, const vec& lambda, chuck_type* residual_constraints=NULL );

    /// Jacobian of the residual
    void compute_jacobian( SolverOperations sop );

    /// perform a line search, i.e. finds the sub-step that decreased "sufficiently" the error
    /// re-arranged from numerical recipies, look at the book for more details!
    bool lnsrch( SReal& resnorm, vec& p /*correction*/, vec& residual, SReal stpmax, SolverOperations sop, core::MultiVecDerivId err, core::MultiVecCoordId newX, const core::MultiVecDerivId newV, core::MultiVecDerivId newF, core::MultiVecCoordId oldX, core::MultiVecDerivId oldV, core::MultiVecDerivId oldF );


    /// multivec temporaries
    core::behavior::MultiVecCoord _x0; ///< position at the beginning of the time step
    core::behavior::MultiVecDeriv _v0; ///< velocity at the beginning of the time step
    core::behavior::MultiVecDeriv _f0; ///< force at the beginning of the time step - TODO do not allocate it if alpha=1
    core::behavior::MultiVecDeriv _vfirstguess; ///< velocity computed by the first guess (fail-safe solution)
    core::behavior::MultiVecDeriv _lambdafirstguess; ///< lambdas computed by the first guess (fail-safe solution)
    core::behavior::MultiVecDeriv _err; ///< residual
    core::behavior::MultiVecDeriv _deltaV; ///< newV - v0

    /// multivec copy including mapped dofs
    void v_eq_all(const core::ExecParams* params, sofa::core::MultiVecId v, sofa::core::ConstMultiVecId a);

    /// keep an eye on which constraints are not bilateral
    helper::vector<linearsolver::Constraint::SPtr> m_projectors;
    /// handle unilateral constraints as an active set of bilateral constraints
    /// so UnilateralProjector are removed from the system so the solver solve their correction as bilateral
    /// note the first guess always treat them as unilateral
    /// during non-linear correction passes, if the unilateral constraint is not violated, it does not generate error that does not generate correction
    void handleUnilateralConstraints();

};

}
}
}



#endif
