#ifndef COMPLIANT_ASSEMBLEDSOLVER_H
#define COMPLIANT_ASSEMBLEDSOLVER_H


#include <Compliant/config.h>

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/MultiVec.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>

#include <Compliant/assembly/AssembledSystem.h>

#include <sofa/helper/OptionsGroup.h>



namespace sofa {

namespace simulation {
class AssemblyVisitor;

namespace common {
class MechanicalOperations;
class VectorOperations;
}

}

namespace component {

namespace linearsolver {
class AssembledSystem;
class KKTSolver;
}


namespace odesolver {
			



/** DAE Solver combining implicit time integration and constraint stabilization.
 * The system matrix is assembled in a regularized KKT form.
  Constraint compliance is used to regularize the equation, so that the Schur complement is always positive definite.

  Inspired from Servin,Lacoursiere,Melin, Interactive Simulation of Elastic Deformable Materials,  http://www.ep.liu.se/ecp/019/005/ecp01905.pdf
  We generalize it to a tunable implicit integration scheme:
      \f[ \begin{array}{ccc}
    \Delta v &=& h.M^{-1}.(\alpha f_{n+1} + (1-\alpha) f_n)  \\
    \Delta x &=& h.(\beta v_{n+1} + (1-\beta) v_n)
    \end{array} \f]
    where \f$ h \f$ is the time step, \f$ \alpha \f$ is the implicit velocity factor, and \f$ \beta \f$ is the implicit position factor.

    The corresponding dynamic equation is:
  \f[ \left( \begin{array}{cc} \frac{1}{h} PM & -PJ^T \\
                               J & \frac{1}{l} C \end{array}\right)
      \left( \begin{array}{c} \Delta v \\ \bar\lambda \end{array}\right)
    = \left( \begin{array}{c} Pf \\ - \frac{1}{l} (\phi +(d+\alpha h) \dot \phi)  \end{array}\right) \f]
    where \f$ M \f$ is the mass matrix, \f$ P \f$ is a projection matrix to impose boundary conditions on displacements (typically maintain fixed points), \f$ \phi \f$ is the constraint violation, \f$ J \f$ the constraint Jacobian matrix,
    \f$ C \f$ is the compliance matrix (i.e. inverse of constraint stiffness) used to soften the constraints, \f$ l=\alpha(h \beta + d) \f$ is a term related to implicit integration and constraint damping, and
      \f$ \bar\lambda \f$ is the average constraint forces, consistently with the implicit velocity integration.

      The system is singular due to the projection matrix \f$ P \f$ (corresponding to the projective constraints applied to the independent DOFs), however we can use \f$ P M^{-1}P \f$ as inverse mass matrix to compute Schur complements.

 In the default implementation, a Schur complement is used to compute the constraint forces, then these are added to the external forces to obtain the final velocity increment,
  and the positions are updated according to the implicit scheme:

  \f[ \begin{array}{ccc}
   ( hJPM^{-1}PJ^T + \frac{1}{l}C ) \bar\lambda &=& -\frac{1}{l} (\phi + (d+h\alpha)\dot\phi ) - h J M^{-1} f \\
                                 \Delta v  &=&  h P M^{-1}( f + J^T \bar\lambda ) \\
                                 \Delta x  &=&  h( v + \beta \Delta v )
  \end{array} \f]


  A word on Rayleigh damping:
  It is not handled at the solver level (contrarly to ImplicitEulerSolver) in order not to bias the equation.
  It can be added directly from the ForceFields and Masses components.
  Note that in that case, the Rayleigh damping does NOT consider the geometric stiffnesses.
  It could be possible to bias the child force used to compute the geometric stiffness but it would imposed to each forcefield to compute a weighted "rayleigh force" in addition to the regular force. It is neglicted for now.

 @author Francois Faure, Maxime Tournier, Matthieu Nesme
*/
class SOFA_Compliant_API CompliantImplicitSolver : public sofa::core::behavior::OdeSolver {

    public:

    /** Unification of the parameters and helpers used by the solver */
    struct SolverOperations
    {
        sofa::simulation::common::VectorOperations vop;
        sofa::simulation::common::MechanicalOperations mop;
        sofa::core::objectmodel::BaseContext* ctx;
        SReal alpha;
        SReal beta;
        core::MechanicalParams _mparams;
        core::MultiVecCoordId posId;
        core::MultiVecDerivId velId;

        SolverOperations( const core::ExecParams* ep , sofa::core::objectmodel::BaseContext* ctx,
                          SReal a, SReal b, SReal dt,
                          const core::MultiVecCoordId& posId, const core::MultiVecDerivId& velId,
                          bool precomputedTraversalOrder = false,
                          bool staticSolver=false )
            : vop( ep, ctx, precomputedTraversalOrder )
            , mop( ep, ctx, precomputedTraversalOrder )
            , ctx( ctx )
            , alpha( a )
            , beta( b )
            , posId( posId )
            , velId( velId )
        {
            _mparams.setExecParams( ep );
            mop->setImplicit(true); // we will compute forces and stiffnesses

            SReal mfactor;
            SReal bfactor;
            SReal kfactor;

            if( staticSolver )
            {
                mfactor = 0.0;
                bfactor = 0.0;
                kfactor = 1.0;
                dt = 1;
            }
            else
            {
                mfactor = 1.0;
                bfactor = -dt * alpha;
                kfactor = -dt * dt * alpha * beta;
            }

            mparams().setMFactor( mfactor );
            mparams().setBFactor( bfactor );
            mparams().setKFactor( kfactor );
            mparams().setDt( dt );

            mparams().setImplicitVelocity( alpha );
            mparams().setImplicitPosition( beta );

//            mparams().setX( posId );
//            mparams().setV( velId );

            mop.mparams = mparams();
        }

//        SolverOperations( const SolverOperations& sop )
//            : vop(sop.vop)
//            , mop(sop.mop)
//            , ctx(sop.ctx)
//            , alpha(sop.alpha)
//            , beta(sop.beta)
//            , _mparams(sop._mparams)
//            , posId(sop.posId)
//            ,velId(sop.velId)
//        {}

        inline const core::MechanicalParams& mparams() const { return /*mop.*/_mparams; }
        inline       core::MechanicalParams& mparams()       { return /*mop.*/_mparams; }

    };


				
	SOFA_CLASS(CompliantImplicitSolver, sofa::core::behavior::OdeSolver);


    typedef linearsolver::AssembledSystem system_type;
				
    virtual void init();
    virtual void parse(core::objectmodel::BaseObjectDescription* arg);

    // OdeSolver API
    virtual void solve(const core::ExecParams* params,
                       SReal dt,
                       core::MultiVecCoordId posId,
                       core::MultiVecDerivId velId);


	CompliantImplicitSolver();
    virtual ~CompliantImplicitSolver();

    virtual void reset();
    virtual void cleanup();

    enum { NO_STABILIZATION=0, PRE_STABILIZATION, POST_STABILIZATION_RHS, POST_STABILIZATION_ASSEMBLY, NB_STABILIZATION };
    Data<helper::OptionsGroup> stabilization;

    Data<bool> warm_start;
    Data<bool> debug;
    Data<helper::OptionsGroup> constraint_forces;
    Data<SReal> alpha;     ///< the \alpha and \beta parameters of the integration scheme
    Data<SReal> beta;     ///< the \alpha and \beta parameters of the integration scheme
	Data<SReal> stabilization_damping;

    enum { FORMULATION_VEL=0, FORMULATION_DV, FORMULATION_ACC, NB_FORMULATION };
    Data<helper::OptionsGroup> formulation;

    Data<bool> neglecting_compliance_forces_in_geometric_stiffness; ///< isn't the name clear enough?


  protected:

    // keep a pointer on the visitor used to assemble
    simulation::AssemblyVisitor *assemblyVisitor;

    /// a derivable function creating and calling the assembly visitor to create an AssembledSystem
    virtual void perform_assembly( const core::MechanicalParams *mparams, system_type& sys );
				
	// send a visitor 
    void send(simulation::Visitor& vis, bool precomputedTraversalOrder=true);
			  
	// integrate positions
    virtual void integrate( SolverOperations& sop,
                            const core::MultiVecCoordId& posId,
                            const core::MultiVecDerivId& velId );

    // integrate positions and velocities
    virtual void integrate( SolverOperations& sop,
                            const core::MultiVecCoordId& posId,
                            const core::MultiVecDerivId& velId,
                            const core::MultiVecDerivId& accId,
                            const SReal& accFactor );

	// propagate velocities
    void propagate(const core::MechanicalParams* params);
	
	// linear solver: TODO hide in pimpl ?
	typedef linearsolver::KKTSolver kkt_type;
    core::sptr<kkt_type> kkt;




	// TODO: hide 
public:
	typedef system_type::vec vec;


    /// Compute the forces f (stiffness and constraint forces)
    virtual void compute_forces(SolverOperations& sop,
                                core::behavior::MultiVecDeriv& f,  // the total force sum (stiffness + constraint forces if required)
                                core::behavior::MultiVecDeriv* f_k = NULL // the stiffness force only
                               );

    /// evaluate violated and active constraints
    void filter_constraints(const core::MultiVecCoordId& posId) const;
    /// reset violated and active constraints
    void clear_constraints() const;



    /// linear rhs (ode & constraints) for dynamics steps
    /// the right part of the implicit system c (c_k in compliant-reference.pdf, section 3)
    /// f_k(in) must contain the stiffness forces, and (out) it will contains c_k
    virtual void rhs_dynamics(SolverOperations& sop,
                              vec& res, const system_type& sys,
                              core::behavior::MultiVecDeriv& f_k,
                              core::MultiVecCoordId posId,
                              core::MultiVecDerivId velId ) const;

    /// linear rhs (constraints only) for dynamics steps
    virtual void rhs_constraints_dynamics(vec& res, const system_type& sys,
                                          core::MultiVecCoordId posId,
                                          core::MultiVecDerivId velId ) const;

    /// linear rhs (ode & constraints) for correction step
    virtual void rhs_correction(vec& res, const system_type& sys,
                                core::MultiVecCoordId posId,
                                core::MultiVecDerivId velId) const;
	
	// current v, lambda
    virtual void get_state(vec& res, const system_type& sys, const core::MultiVecDerivId& multiVecId) const;

	// set v, lambda
    virtual void set_state(const system_type& sys, const vec& data, const core::MultiVecDerivId& multiVecId) const;
    // set v
    virtual void set_state_v(const system_type& sys, const vec& data, const core::MultiVecDerivId& multiVecId) const;
    // set lambda
    virtual void set_state_lambda(const system_type& sys, const vec& data) const;


	// this is for warm start and returning constraint forces
    core::behavior::MultiVecDeriv lagrange;

    /** @group Debug and unit testing */
    //@{
    // toggle the recording of the dynamics vectors
    void storeDynamicsSolution(bool); ///< if true, store the dv and lambda at each dynamics solution
    // right-hand side (f,phi) and unknow (dv,lambda) of the dynamics system:
    system_type::vec getLambda() const { assert(storeDSol); return dynamics_solution.tail(sys.n); }
    system_type::vec getDv() const { assert(storeDSol); return dynamics_solution.head(sys.m); }
    system_type::vec getPhi() const { assert(storeDSol); return dynamics_rhs.tail(sys.n); }
//    system_type::vec getF() const { assert(storeDSol); return dynamics_rhs.head(sys.m); }  FF: I suspect this one is wrong, because rhs does not contain forces but momenta, does it ?
    // assembled matrices
    const system_type::rmat& H() const {return sys.H;}
    const system_type::rmat& P() const {return sys.P;}
    const system_type::rmat& J() const {return sys.J;}
    const system_type::rmat& C() const {return sys.C;}
    //@}

    /// compute post-stabilization correcting constraint in position-based
    /// have a look to Ascher94&97 and Cline03 for more details
    /// if fullAssembly==true, a complete assembly/factorization is performed
    /// otherwise only the rhs is updated and the system used for the dynamics pass is preserved
    /// @warning: the contacts points and normals are not updated, so the time step needs to be small with contacts
    virtual void post_stabilization( SolverOperations& sop,
                             core::MultiVecCoordId posId, core::MultiVecDerivId velId,
                             bool fullAssembly, bool realloc=false );

protected:

    system_type sys; ///< assembled equation system
    bool storeDSol;
    vec dynamics_solution;       ///< to store dv and lambda
    vec dynamics_rhs;            ///< to store f and phi, the right-hand side


    /// temporary multivecs
    core::behavior::MultiVecDeriv _ck; ///< the right part of the implicit system (c_k term)
    core::behavior::MultiVecDeriv _acc; ///< acceleration when FORMULATION_ACC, or dv when FORMULATION_DV


};

}
}
}



#endif
