#ifndef COMPLIANTDEV_ASSEMBLEDSOLVER_H
#define COMPLIANTDEV_ASSEMBLEDSOLVER_H


#include "initCompliant.h"

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/MultiVec.h>
#include <sofa/simulation/common/MechanicalVisitor.h>

// TODO forward instead ?
#include "KKTSolver.h"

namespace sofa {

namespace simulation {
struct AssemblyVisitor;
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

\sa \ref sofa::core::behavior::BaseCompliance

*/


class SOFA_Compliant_API AssembledSolver : public sofa::core::behavior::OdeSolver {
  public:
				
	SOFA_CLASS(AssembledSolver, sofa::core::behavior::OdeSolver);
				
    virtual void init();

    virtual void solve(const core::ExecParams* params,
	                   double dt, 
                       core::MultiVecCoordId posId,
                       core::MultiVecDerivId velId,
                       bool computeForce, // should the right part of the implicit system be computed?
                       bool integratePosition, // should the position be updated?
                       simulation::AssemblyVisitor *vis
                       );

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
                       core::MultiVecDerivId velId
                       )
    {
        solve( params, dt, posId, velId, true, true );
    }


	AssembledSolver();
	~AssembledSolver();
	
	virtual void cleanup();

    // mechanical params
    virtual core::MechanicalParams mparams(const core::ExecParams& params,
                                           double dt) const;

    // solve velocity dynamics ?
	Data<bool> use_velocity, warm_start, propagate_lambdas, stabilization, debug;
    Data<SReal> f_rayleighStiffness, f_rayleighMass;  ///< uniform Rayleigh damping ratio applied to the stiffness and mass matrices
    Data<SReal> implicitVelocity, implicitPosition;     ///< the \f$ \alpha \f$ and \f$ \beta  \f$ parameters of the integration scheme

    simulation::AssemblyVisitor* _assemblyVisitor;


	
  protected:
				
	// send a visitor 
	void send(simulation::Visitor& vis);
				
	// integrate positions
    void integrate( const core::MechanicalParams* params, core::MultiVecCoordId posId, core::MultiVecDerivId velId );
    // integrate positions and velocities
    void integrate( const core::MechanicalParams* params, core::MultiVecCoordId posId, core::MultiVecDerivId velId, core::MultiVecDerivId dvId );


	// propagate velocities
	void propagate(const core::MechanicalParams* params);	
				

	// linear solver: TODO hide in pimpl ?
	typedef linearsolver::KKTSolver kkt_type;
	kkt_type::SPtr kkt;

public:


    // compute forces
    void forces(const core::MechanicalParams& params);

	typedef linearsolver::AssembledSystem system_type;
	// obtain linear system rhs from system 
    virtual kkt_type::vec rhs(const system_type& sys, bool computeForce = true) const;

	// warm start solution
	virtual kkt_type::vec warm(const system_type& sys) const;


	// velocities from system solution 
	virtual kkt_type::vec velocity(const system_type& sys, const kkt_type::vec& x) const;

	// constraint forces from system solution
	virtual kkt_type::vec lambda(const system_type& sys, const kkt_type::vec& x) const;

	// mask for constraints to be stabilized
	kkt_type::vec stab_mask(const system_type& sys) const;

	// this is for warm start and returning constraint forces
	core::behavior::MultiVecDeriv lagrange;


protected:

	void alloc(const core::ExecParams& params);


    struct propagate_visitor : simulation::MechanicalVisitor {

        core::MultiVecDerivId out, in;

        propagate_visitor(const sofa::core::MechanicalParams* mparams) : simulation::MechanicalVisitor(mparams) { }

        Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm) {
            // clear dst
            mm->resetForce(this->params /* PARAMS FIRST */, out.getId(mm));
            return RESULT_CONTINUE;
        }

        void bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map) {
            map->applyJT(mparams /* PARAMS FIRST */, out, in);
        }

    };

};

}
}
}



#endif
