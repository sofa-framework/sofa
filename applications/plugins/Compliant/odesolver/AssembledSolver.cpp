#include "AssembledSolver.h"

#include <sofa/component/linearsolver/EigenSparseMatrix.h>
#include <sofa/component/linearsolver/EigenVector.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/MechanicalOperations.h>
#include <sofa/simulation/common/VectorOperations.h>

#include "assembly/AssemblyVisitor.h"
#include "misc/Stabilization.h"

#include "utils/minres.h"
#include "utils/scoped.h"



#ifdef GR_BENCHMARK
    #include <sofa/helper/system/thread/CTime.h>
#endif


namespace sofa {
namespace component {
namespace odesolver {

SOFA_DECL_CLASS(AssembledSolver);
int AssembledSolverClass = core::RegisterObject("Example compliance solver using assembly").add< AssembledSolver >();

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace core::behavior;


AssembledSolver::AssembledSolver()
	: warm_start(initData(&warm_start, 
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
	                 "print debug stuff")),
// god dammit i said no rayleigh mass/stiffness crap !!11
	f_rayleighStiffness( initData(&f_rayleighStiffness,
	                              SReal(0),
	                                "rayleighStiffness",
	                                "Rayleigh damping coefficient related to stiffness, >= 0") ),
	f_rayleighMass( initData(&f_rayleighMass,
	                         SReal(0),
	                         "rayleighMass",
	                         "Rayleigh damping coefficient related to mass, >= 0")),
	implicitVelocity( initData(&implicitVelocity,SReal(1),
	                           "implicitVelocity",
	                           "Weight of the next forces in the average forces used to update the velocities. 1 is implicit, 0 is explicit.")),
	implicitPosition( initData(&implicitPosition,SReal(1),
	                           "implicitPosition",
	                           "Weight of the next velocities in the average velocities used to update the positions. 1 is implicit, 0 is explicit."))
	
{
	
}

void AssembledSolver::send(simulation::Visitor& vis) {
	scoped::timer step("visitor execution");
				
	this->getContext()->executeVisitor( &vis );
}
			
		
void AssembledSolver::integrate( const core::MechanicalParams* params,
                                 core::MultiVecCoordId posId,
                                 core::MultiVecDerivId velId ) {
	scoped::timer step("position integration");
	SReal dt = params->dt();
				
	// integrate positions
	sofa::simulation::common::VectorOperations vop( params, this->getContext() );
				
	typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
	VMultiOp multi;
				
	multi.resize(1);
				
    multi[0].first = posId;
    multi[0].second.push_back( std::make_pair(posId, 1.0) );
    multi[0].second.push_back( std::make_pair(velId, dt) );

	vop.v_multiop( multi );
}


void AssembledSolver::integrate( const core::MechanicalParams* params,
                                 core::MultiVecCoordId posId,
                                 core::MultiVecDerivId velId,
                                 core::MultiVecDerivId dvId ) {
    scoped::timer step("position integration");
    SReal dt = params->dt();

    // integrate positions and velocities
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );

    typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
    VMultiOp multi;

    multi.resize(2);

    multi[0].first = posId;
    multi[0].second.push_back( std::make_pair(posId, 1.0) );
    multi[0].second.push_back( std::make_pair(velId, dt) );
    multi[0].second.push_back( std::make_pair(dvId, dt*implicitPosition.getValue()) );


    multi[1].first = velId;
    multi[1].second.push_back( std::make_pair(velId, 1.0) );
    multi[1].second.push_back( std::make_pair(dvId, 1.0) );

    vop.v_multiop( multi );
}


void AssembledSolver::alloc(const core::ExecParams& params) {
	scoped::timer step("lambdas alloc");
	sofa::simulation::common::VectorOperations vop( &params, this->getContext() );
    lagrange.realloc( &vop, false, true );
}

AssembledSolver::~AssembledSolver() {
}

void AssembledSolver::cleanup() {
	sofa::simulation::common::VectorOperations vop( core::ExecParams::defaultInstance(), this->getContext() );
    vop.v_free( lagrange.id(), false, true );
}

		
void AssembledSolver::forces(const core::MechanicalParams& params) {
	scoped::timer step("forces computation");
				
	sofa::simulation::common::MechanicalOperations mop( &params, this->getContext() );
	sofa::simulation::common::VectorOperations vop( &params, this->getContext() );
				
	MultiVecDeriv f  (&vop, core::VecDerivId::force() );
				
    mop.computeForceNeglectingCompliance(f); // f = f0

    f *= params.dt();  // f = h f0
    mop.addMBKv( f, 1, 0, 0 ); // f = Mv + h.f0 = p + h.f0

}


void AssembledSolver::propagate(const core::MechanicalParams* params) {
	simulation::MechanicalPropagatePositionAndVelocityVisitor bob( params );
	send( bob );
}			


void AssembledSolver::buildMparams(core::MechanicalParams& mparams,
                                   core::MechanicalParams& mparamsWithoutStiffness,
                                   const core::ExecParams& params,
                                   double dt) const
{
    // H = (1+h*rm)M - h*B - h*(h+rs)K
    // Rayleigh damping mass factor rm is used with a negative sign because it
    // is recorded as a positive real, while its force is opposed to the velocity

    mparams.setExecParams( &params );
    mparams.setMFactor( 1.0 + f_rayleighMass.getValue() * dt );
    mparams.setBFactor( -dt );
    mparams.setKFactor( -dt * ( dt + f_rayleighStiffness.getValue() ) );
    mparams.setDt( dt );
    mparams.setImplicitVelocity( implicitVelocity.getValue() );
    mparams.setImplicitPosition( implicitPosition.getValue() );


    // to treat compliant components (ie stiffness components treated as compliance), Kfactor must only consider Rayleigh damping part
    mparamsWithoutStiffness.setExecParams( &params );
    mparamsWithoutStiffness.setMFactor( 1.0 + f_rayleighMass.getValue() * dt );
    mparamsWithoutStiffness.setBFactor( -dt );
    mparamsWithoutStiffness.setKFactor( -dt * f_rayleighStiffness.getValue() ); // only the factor relative to Rayleigh damping
    mparamsWithoutStiffness.setDt( dt );
    mparamsWithoutStiffness.setImplicitVelocity( implicitVelocity.getValue() );
    mparamsWithoutStiffness.setImplicitPosition( implicitPosition.getValue() );

}
			
// implicit euler
linearsolver::KKTSolver::vec AssembledSolver::rhs(const system_type& sys, bool computeForce) const {
	kkt_type::vec res = kkt_type::vec::Zero( sys.size() );

    if( !computeForce ) res.head(sys.m).setZero();
    else res.head( sys.m ) = sys.P * sys.b;
	
	if( sys.n ) {

		// remove velocity part from rhs as it's handled implicitly here
		// (this is due to weird Compliance API, TODO fix this)
		res.tail( sys.n ) =  sys.phi + sys.J * sys.v;
		
	}

	return res;
}

linearsolver::KKTSolver::vec AssembledSolver::warm(const system_type& sys) const {
	
	kkt_type::vec res = kkt_type::vec::Zero( sys.size() );

	if( warm_start.getValue() ) {
		// note: warm starting velocities is somehow equivalent to zero
		// acceleration start.

		// velocity
		res.head( sys.m ) = sys.P * sys.v;
		
		// lambdas
		if( sys.n ) res.tail( sys.n ) = sys.dt * sys.lambda;
	}

	return res;	
}


// TODO remove (useless now that use_velocity is gone)
AssembledSolver::kkt_type::vec AssembledSolver::velocity(const system_type& sys,
                                                         const kkt_type::vec& x) const {
	return sys.P * x.head( sys.m );
}

// this is a *force* TODO remove ?
AssembledSolver::kkt_type::vec AssembledSolver::lambda(const system_type& sys,
                                                       const kkt_type::vec& x) const {
	return x.tail( sys.n ) / sys.dt;
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


void AssembledSolver::solve(const core::ExecParams* params,
                   double dt,
                   core::MultiVecCoordId posId,
                   core::MultiVecDerivId velId,
                   bool computeForce, // should the right part of the implicit system be computed?
                   bool integratePosition // should the position be updated?
                   )
{
    // assembly visitor
    core::MechanicalParams mparams, mparamsWithoutStiffness;
    this->buildMparams( mparams, mparamsWithoutStiffness, *params, dt );
    simulation::AssemblyVisitor* v = new simulation::AssemblyVisitor(&mparams, &mparamsWithoutStiffness, velId, lagrange.id()/*, f_rayleighStiffness.getValue(), f_rayleighMass.getValue()*/ );
    solve( params, dt, posId, velId, computeForce, integratePosition, v );
    delete v;
}


void AssembledSolver::solve(const core::ExecParams* params,
                            double /*dt*/,
                            sofa::core::MultiVecCoordId posId,
                            sofa::core::MultiVecDerivId velId,
                            bool computeForce,
                            bool integratePosition,simulation::AssemblyVisitor * v) {
    assert(kkt);

	// obtain mparams
    const core::MechanicalParams *mparams = v->mparams;
				
	// compute forces
    if( computeForce ) forces( *mparams );

     // assembly visitor (keep an accessible pointer from another component all along the solving)
     _assemblyVisitor = v;

	// fetch data
    send( *_assemblyVisitor );
	
	typedef linearsolver::AssembledSystem system_type;
	
	system_type sys = _assemblyVisitor->assemble();
	
	if( debug.getValue() ){
		sys.debug();
	}
	
	// solution vector
	system_type::vec x = warm( sys );

	{
//		scoped::timer step("system factor");
		kkt->factor( sys );
	}

	{
//		scoped::timer step("system solve");





        system_type::vec b = rhs( sys, computeForce );

        // stab mask
        system_type::vec mask = stab_mask( sys );

        if( mask.size() ) {
            scoped::timer step("stabilization");

            // stabilization
            system_type::vec tmp_b = system_type::vec::Zero(sys.size());
            tmp_b.tail( sys.n ) = b.tail(sys.n).array() * mask.array();

            system_type::vec tmp_x = system_type::vec::Zero( sys.size() );

            kkt->solve(tmp_x, sys, tmp_b);

            _assemblyVisitor->distribute_master( velId, velocity(sys, tmp_x) );

            if( integratePosition ) integrate( mparams, posId, velId );

            // zero stabilized constraints
            b.tail(sys.n).array() *= 1 - mask.array();
        }
        
        {
//	        scoped::timer step("dynamics");
	        kkt->solve(x, sys, b);
        }


#ifdef GR_BENCHMARK
        t = sofa::helper::system::thread::CTime::getFastTime()-t;
        std::cerr<<sys.n<<"\t"<<kkt->nbiterations+1<<"\t"<<((double)t)/((double)tfreq)<<std::endl;
#endif
	}
	

	
	if( sys.n ) {
        _assemblyVisitor->distribute_compliant( lagrange.id(), lambda(sys, x) );

		if( propagate_lambdas.getValue() ) {
//			scoped::timer step("lambdas propagation");
            propagate_visitor prop( mparams );
			prop.out = core::VecId::force();
			prop.in = lagrange.id();
			
			send( prop );
		}
	}

	// distribute (projected) velocities
	_assemblyVisitor->distribute_master( velId, velocity(sys, x) );

	// TODO expose logical parts of solving as prepare, dynamics/correction, integrate
	// then remove integratePosition flag

	// update positions 
	if( integratePosition ) integrate( mparams, posId, velId );
}

			
			
void AssembledSolver::init() {
				
	// do want KKTSolver !
	kkt = this->getContext()->get<kkt_type>();
	
	// TODO slightly less dramatic error
	if( !kkt ) throw std::logic_error("AssembledSolver needs a KKTSolver lol");

}



}
}
}
