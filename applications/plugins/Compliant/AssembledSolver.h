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
			
// simple example compliance solver using system assembly,
// integration using velocity implicit euler. needs a KKTSolver
// at the same level in the scene graph.
class SOFA_Compliant_API AssembledSolver : public sofa::core::behavior::OdeSolver {
  public:
				
	SOFA_CLASS(AssembledSolver, sofa::core::behavior::OdeSolver);
				
    virtual void init();

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

    // solve velocity dynamics ?
	Data<bool> use_velocity, warm_start, propagate_lambdas, stabilization, debug;
	
  protected:
				
	// send a visitor 
	void send(simulation::Visitor& vis);
				
	// integrate positions
    void integrate( const core::MechanicalParams* params, core::MultiVecCoordId posId, core::MultiVecDerivId velId );
				
	// compute forces
	void forces(const core::ExecParams& params);

	// propagate velocities
	void propagate(const core::MechanicalParams* params);
	
	// mechanical params
	virtual core::MechanicalParams mparams(const core::ExecParams& params, 
	                                       double dt) const;
				

	// linear solver: TODO hide in pimpl ?
	typedef linearsolver::KKTSolver kkt_type;
	kkt_type::SPtr kkt;

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
