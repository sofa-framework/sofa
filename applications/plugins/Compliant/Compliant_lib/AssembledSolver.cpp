#include "AssembledSolver.h"

#include <sofa/component/linearsolver/EigenSparseMatrix.h>
#include <sofa/component/linearsolver/EigenVector.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/MechanicalOperations.h>
#include <sofa/simulation/common/VectorOperations.h>

#include "AssemblyVisitor.h"
#include "Stabilization.h"

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
	                      "warm start iterative solvers: avoids biasing solution towards zero (and speeds-up resolution)")),
	  propagate_lambdas(initData(&propagate_lambdas, 
	                             false,
	                             "propagate_lambdas",
	                             "propagate Lagrange multipliers in force vector at the end of time step")),
	  stabilization(initData(&stabilization, 
	                         false,
	                         "stabilization",
	                         "apply a stabilization pass on kinematic constraints requesting it")),
	debug(initData(&debug, 
	               false,
	               "debug",
	               "print debug stuff"))
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
    lagrange.realloc( &vop );
}

AssembledSolver::~AssembledSolver() {
}

void AssembledSolver::cleanup() {
	sofa::simulation::common::VectorOperations vop( core::ExecParams::defaultInstance(), this->getContext() );
	vop.v_free( lagrange.id(), false );
}

		
void AssembledSolver::forces(const core::ExecParams& params) {
	scoped::timer step("forces computation");
				
	sofa::simulation::common::MechanicalOperations mop( &params, this->getContext() );
	sofa::simulation::common::VectorOperations vop( &params, this->getContext() );
				
	MultiVecDeriv f  (&vop, core::VecDerivId::force() );
				
    mop.computeForceNeglectingCompliance(f);
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
    res.setKFactor( 1.0 ); // not necessary but just in case it is used somewhere
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
		if( use_velocity.getValue() ) {
			
			res.head( sys.m ) = sys.P * sys.v;
			if( sys.n ) res.tail( sys.n ) = sys.dt * sys.lambda;
			
		} else {
			if( sys.n ) res.tail( sys.n ) = sys.lambda;
		}
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

// this is a force
AssembledSolver::kkt_type::vec AssembledSolver::lambda(const system_type& sys,
                                                       const kkt_type::vec& x) const {
	if( use_velocity.getValue() ) return x.tail( sys.n ) / sys.dt;
	else return x.tail( sys.n );

}


AssembledSolver::kkt_type::vec AssembledSolver::stab_mask(const system_type& sys) const {
	kkt_type::vec res;

	if( !sys.n ) return res;
	if( !stabilization.getValue() ) return res;

	res = kkt_type::vec::Zero( sys.n );
	
	unsigned off = 0;

	bool none = true;
	
	for(unsigned i = 0, n = sys.compliant.size(); i < n; ++i) {
		
		system_type::dofs_type* const dofs = sys.compliant[i];
		unsigned dim = dofs->getMatrixSize();

		Stabilization* post = dofs->getContext()->get<Stabilization>(core::objectmodel::BaseContext::Local);
				
		if( post ) {

            if( post->mask.size() == dim )
            {
                for( unsigned j=0 ; j<dim ; ++j )
                    res[off+j] = post->mask[j] ? (SReal)1.0 : (SReal)0.0;
            }
            else
            {
                res.segment( off, dim ).setOnes();
            }
			none = false;
		}
		
		off += dim;
	}
	
	// empty vec
	if( none ) res.resize( 0 );
	
	return res;
}


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
	
	// TODO do this inside visitor ctor instead
	vis.lagrange = lagrange.id();
	
	// fetch data
	send( vis );
	
	typedef linearsolver::AssembledSystem system_type;
	
	sofa::helper::AdvancedTimer::stepBegin( "assembly" );
	system_type sys = vis.assemble();
	sofa::helper::AdvancedTimer::stepEnd( "assembly" );
	
	if( debug.getValue() ){
		sys.debug();
	}
	
	// solution vector
	system_type::vec x = warm( sys );

	// stab mask
	system_type::vec mask = stab_mask( sys );
	
	{
		scoped::timer step("system factor");
		kkt->factor( sys );
	}

	{
		scoped::timer step("system solve");
	
		system_type::vec b = rhs(sys);
		
		if( mask.size() ) {
			scoped::timer step("stabilization");
			assert( use_velocity.getValue() && "does not work on acceleration !");
			
			// stabilization
			system_type::vec tmp_b = system_type::vec::Zero(sys.size());
			tmp_b.tail( sys.n ) = b.tail(sys.n).array() * mask.array();
			
			system_type::vec tmp_x = system_type::vec::Zero( sys.size() );
			
			kkt->solve(tmp_x, sys, tmp_b);
			vis.distribute_master( core::VecId::velocity(), velocity(sys, tmp_x) );
			integrate( &mparams );
			
			// zero stabilized constraints
			b.tail(sys.n).array() *= 1 - mask.array();
		}
		
		kkt->solve(x, sys, b);
	}
	
	
	// distribute (projected) velocities
	vis.distribute_master( core::VecId::velocity(), velocity(sys, x) );
	
	if( sys.n ) {
		vis.distribute_compliant( lagrange.id(), lambda(sys, x) );

		if( propagate_lambdas.getValue() ) {
			scoped::timer step("lambdas propagation");
			propagate_visitor prop( &mparams );
			prop.out = core::VecId::force();
			prop.in = lagrange.id();
			
			send( prop );
		}
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
