#ifndef COMPLIANT_ASSEMBLEDSOLVER_H
#define COMPLIANT_ASSEMBLEDSOLVER_H


#include "initCompliant.h"

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/MultiVec.h>
#include <sofa/simulation/common/MechanicalVisitor.h>

// TODO forward instead ?
#include "numericalsolver/KKTSolver.h"

namespace sofa {

namespace simulation {
struct AssemblyVisitor;

namespace common {
class MechanicalOperations;
class VectorOperations;
}

}

namespace component {

namespace linearsolver {
struct AssembledSystem;
}


namespace odesolver {
			


/** Solver using a regularized KKT matrix.
  The compliance values are used to regularize the system.

 simple example compliance solver using system assembly,
 integration using velocity implicit euler. needs a KKTSolver
 at the same level in the scene graph.


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
  It is not handled at the solver level (contrarly to ImplicitEulerSolver) in order not to pollutate. It can be added directly from the ForceFields and Masses components.
  Note that in that case, the Rayleigh damping does NOT consider the geometric stiffnesses.
  It could be possible to bias the child force used to compute the geometric stiffness but it would imposed to each forcefield to compute a weighted "rayleigh force" in addition to the regular force. It is neglicted for now.
*/


class SOFA_Compliant_API AssembledSolver : public sofa::core::behavior::OdeSolver {
  public:
				
	SOFA_CLASS(AssembledSolver, sofa::core::behavior::OdeSolver);


    typedef linearsolver::AssembledSystem system_type;
				
    virtual void init();

	// broken, will go soon
    virtual void solve(const core::ExecParams* params,
	                   double dt, 
                       core::MultiVecCoordId posId,
                       core::MultiVecDerivId velId,
                       bool computeForce, // should the right part of the implicit system be computed?
                       bool integratePosition, // should the position be updated?
                       simulation::AssemblyVisitor *vis
                       );

	// broken, will go soon
    virtual void solve(const core::ExecParams* params,
                       double dt,
                       core::MultiVecCoordId posId,
                       core::MultiVecDerivId velId,
                       bool computeForce, // should the right part of the implicit system be computed?
                       bool integratePosition // should the position be updated?
                       );

    // OdeSolver API
    virtual void solve(const core::ExecParams* params,
                       double dt,
                       core::MultiVecCoordId posId,
                       core::MultiVecDerivId velId);


	AssembledSolver();
    virtual ~AssembledSolver();
	
	virtual void cleanup();

    // mechanical params
    void buildMparams( core::MechanicalParams& mparams,
                       const core::ExecParams& params,
                       double dt) const;

    Data<bool> warm_start, propagate_lambdas, stabilization, debug;
    Data<SReal> alpha, beta;     ///< the \alpha and \beta parameters of the integration scheme


  protected:

    // keep a pointer on the visitor used to assemble
    simulation::AssemblyVisitor *assemblyVisitor;

    /// a derivable function creating and calling the assembly visitor to create an AssembledSystem
    virtual void perform_assembly( const core::MechanicalParams *mparams, system_type& sys );
				
	// send a visitor 
    void send(simulation::Visitor& vis);
			  
	// integrate positions
    void integrate( const core::MechanicalParams* params, 
					core::MultiVecCoordId posId, 
					core::MultiVecDerivId velId );

	// propagate velocities
    void propagate(const core::MechanicalParams* params);
	
	// linear solver: TODO hide in pimpl ?
	typedef linearsolver::KKTSolver kkt_type;
	kkt_type::SPtr kkt;

	// TODO: hide 
public:
	typedef system_type::vec vec;


    /** Compute the forces f (summing stiffness and compliance) and the right part of the implicit system c (c_k in compliant-reference.pdf, section 3)
      */
    virtual void compute_forces(const core::MechanicalParams& params,
                               simulation::common::MechanicalOperations& mop,
                               simulation::common::VectorOperations& vop,
                               core::behavior::MultiVecDeriv& f,
                               core::behavior::MultiVecDeriv& c );

	// linear rhs for dynamics/correction steps
    virtual void rhs_dynamics(vec& res, const system_type& sys, const vec& v, const core::behavior::MultiVecDeriv& b ) const;
	virtual void rhs_correction(vec& res, const system_type& sys) const;
	
	// current v, lambda
	virtual void get_state(vec& res, const system_type& sys) const;

	// set v, lambda
	virtual void set_state(const system_type& sys, const vec& data) const;


	// TODO does this work yo ?
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
    const system_type::rmat& M() const {return sys.H;}
    const system_type::rmat& P() const {return sys.P;}
    const system_type::rmat& J() const {return sys.J;}
    const system_type::rmat& C() const {return sys.C;}

    //@}

protected:

    system_type sys; ///< assembled equation system
    bool storeDSol;
    vec dynamics_solution;       ///< to store dv and lambda
    vec dynamics_rhs;            ///< to store f and phi, the right-hand side

	void alloc(const core::ExecParams& params);


    struct propagate_visitor : simulation::MechanicalVisitor {

        core::MultiVecDerivId out, in;

        propagate_visitor(const sofa::core::MechanicalParams* mparams);

        Result fwdMappedMechanicalState(simulation::Node* node,
                                        core::behavior::BaseMechanicalState* state);

        Result fwdMechanicalState(simulation::Node* node,
                                core::behavior::BaseMechanicalState* state);

        void bwdMechanicalMapping(simulation::Node* node, core::BaseMapping* map);

    };

};

}
}
}



#endif
