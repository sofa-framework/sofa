#include "AssembledSolver.h"

#include <sofa/component/linearsolver/EigenSparseMatrix.h>
#include <sofa/component/linearsolver/EigenVector.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/MechanicalOperations.h>
#include <sofa/simulation/common/VectorOperations.h>

#include "AssemblyVisitor.h"

#include "utils/minres.h"
#include "utils/scoped.h"

namespace sofa {
namespace component {
namespace odesolver {

SOFA_DECL_CLASS(AssembledSolver);
int AssembledSolverClass = core::RegisterObject("Example compliance solver using assembly").add< AssembledSolver >();

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace core::behavior;


AssembledSolver::AssembledSolver()
	: use_velocity(initData(&use_velocity, 
	                        true,
	                        "use_velocity",
	                        "solve velocity dynamics (otherwise acceleration). this might cause damping when used with iterative solver unless warm_start is on.")),
	  warm_start(initData(&warm_start, 
	                      true,
	                      "warm_start",
	                      "warm start iterative solvers: avoids biasing solution towards zero (and speeds-up resolution)"))
{
	
}

void AssembledSolver::send(simulation::Visitor& vis) {
	scoped::timer step("visitor execution");
				
	this->getContext()->executeVisitor( &vis );
}
			
		
void AssembledSolver::integrate(const core::MechanicalParams* params) {
	scoped::timer step("position integration");
	SReal dt = params->dt();
				
	// integrate positions
	sofa::simulation::common::VectorOperations vop( params, this->getContext() );
	MultiVecCoord pos(&vop, core::VecCoordId::position() );
	MultiVecDeriv vel(&vop, core::VecDerivId::velocity() );
				
	typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
	VMultiOp multi;
				
	multi.resize(1);
				
	multi[0].first = pos.id();
	multi[0].second.push_back( std::make_pair(pos.id(), 1.0) );
	multi[0].second.push_back( std::make_pair(vel.id(), dt) );
				
	vop.v_multiop( multi );
}
		

void AssembledSolver::alloc(const core::ExecParams& params) {
	scoped::timer step("lambdas alloc");
	sofa::simulation::common::VectorOperations vop( &params, this->getContext() );
	
	lagrange.set(&vop);
	vop.v_realloc( lagrange.id(), false );
}

AssembledSolver::~AssembledSolver() {
	// sofa::simulation::common::VectorOperations vop( core::ExecParams::defaultInstance(), this->getContext() );
	// vop.v_free( lagrange.id(), false );
}

		
void AssembledSolver::forces(const core::ExecParams& params) {
	scoped::timer step("forces computation");
				
	sofa::simulation::common::MechanicalOperations mop( &params, this->getContext() );
	sofa::simulation::common::VectorOperations vop( &params, this->getContext() );
				
	MultiVecDeriv f  (&vop, core::VecDerivId::force() );
				
	mop.computeForce(f);
	// mop.projectResponse(f); 
}


void AssembledSolver::propagate(const core::MechanicalParams* params) {
	simulation::MechanicalPropagatePositionAndVelocityVisitor bob( params );
	send( bob );
}			


core::MechanicalParams AssembledSolver::mparams(const core::ExecParams& params, 
                                                double dt) const {
	core::MechanicalParams res( params );
				
	res.setMFactor( 1.0 );
	res.setDt( dt );
				
	res.setImplicitVelocity( 1 );
	res.setImplicitPosition( 1 );

	return res;
}
			
// implicit euler
linearsolver::KKTSolver::vec AssembledSolver::rhs(const system_type& sys) const {

	kkt_type::vec res = kkt_type::vec::Zero( sys.size() );

	if( use_velocity.getValue() ) {
		res.head( sys.m ) = sys.P * (sys.p  +  sys.dt * sys.f);
	} else {
		
    // H = M - h^2 K
		// p = M v
		// hence hKv = 1/h ( p - H v )
		
		kkt_type::vec hKv = (sys.p - (sys.H * sys.v)) / sys.dt;
		res.head( sys.m ) = sys.P * (sys.f + hKv);
	}
	
	if( sys.n ) {

		if( use_velocity.getValue() ) { 
			// remove velocity part from rhs as it's handled implicitly here
			// (this is due to weird Compliance API, TODO fix this)
			res.tail( sys.n ) =  sys.phi + sys.J * sys.v;
		} else {
			res.tail( sys.n ) =  sys.phi / sys.dt;
		}
		
	}

	return res;
}

linearsolver::KKTSolver::vec AssembledSolver::warm(const system_type& sys) const {
	
	kkt_type::vec res = kkt_type::vec::Zero( sys.size() );

	// warm starting is a bad idea anyways
	if( warm_start.getValue() ) {
		if( use_velocity.getValue() ) res.head( sys.m ) = sys.P * sys.v;
	}

	return res;	
}


// AssembledSolver::system_type AssembledSolver::assemble(const simulation::AssemblyVisitor& vis) const {
// 	system_type sys = vis.assemble();
	
// 	// TODO this should be done during system assembly

// 	// fix compliance 
// 	if( sys.n ) {
// 		// std::cerr << "compliance" << std::endl;
		
// 		system_type::vec h = system_type::vec::Constant(sys.n, sys.dt);
// 		system_type::vec fix; fix.resize(sys.n);

// 		fix.array() = 1 / ( h.array() * (h.array() + sys.damping.array()) );
		
// 		sys.C = sys.C * fix.asDiagonal();
// 	}

// 	return sys;
// }
	
AssembledSolver::kkt_type::vec AssembledSolver::velocity(const system_type& sys,
                                                         const kkt_type::vec& x) const {
	if( use_velocity.getValue() ) return sys.P * x.head( sys.m );
	else return sys.P * (sys.v + sys.dt * x.head( sys.m ));
}

AssembledSolver::kkt_type::vec AssembledSolver::lambda(const system_type& sys,
                                                       const kkt_type::vec& x) const {
	if( use_velocity.getValue() ) return x.tail( sys.n ) / sys.dt;
	else return x.tail( sys.n );

}



		
void AssembledSolver::solve(const core::ExecParams* params, 
                            double dt, 
                            sofa::core::MultiVecCoordId , 
                            sofa::core::MultiVecDerivId ) {
	assert(kkt && "i need a kkt solver lol");
				
	// obtain mparams
	core::MechanicalParams mparams = this->mparams(*params, dt);
				
	// compute forces
	forces( mparams );			
				
	// assembly visitor
	simulation::AssemblyVisitor vis(&mparams);
	
	// TODO do this inside visitor ctor
	vis.lagrange = lagrange.id();
	
	// fetch data
	send( vis );
	
	typedef linearsolver::AssembledSystem system_type;
	
	sofa::helper::AdvancedTimer::stepBegin( "assembly" );
	system_type sys = vis.assemble();
	sofa::helper::AdvancedTimer::stepEnd( "assembly" );
	
	// solution vector
	system_type::vec x = warm( sys );
	
	{
		scoped::timer step("system solve");
		kkt->factor( sys );
		kkt->solve(x, sys, rhs( sys ) );
	}
	 
	// distribute (projected) velocities
	vis.distribute_master( core::VecId::velocity(), velocity(sys, x) );
	if( sys.n ) {
		vis.distribute_compliant( lagrange.id(), lambda(sys, x) );
	}
	
	// update positions TODO use xResult/vResult
	integrate( &mparams );
}

			
			
void AssembledSolver::init() {
				
	// let's find a KKTSolver 
	kkt = this->getContext()->get<kkt_type>();
				
	// TODO less dramatic error
	if( !kkt ) throw std::logic_error("AssembledSolver needs a KKTSolver lol");
				
}



}
}
}
