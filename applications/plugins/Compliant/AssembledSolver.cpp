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

    , f_rayleighStiffness( initData(&f_rayleighStiffness,(SReal)0,"rayleighStiffness","Rayleigh damping coefficient related to stiffness, >= 0") )
    , f_rayleighMass( initData(&f_rayleighMass,(SReal)0,"rayleighMass","Rayleigh damping coefficient related to mass, >= 0"))
    , implicitVelocity( initData(&implicitVelocity,(SReal)1.,"implicitVelocity","Weight of the next forces in the average forces used to update the velocities. 1 is implicit, 0 is explicit. (Warning: not relevant if use_velocity=1)"))
    , implicitPosition( initData(&implicitPosition,(SReal)1.,"implicitPosition","Weight of the next velocities in the average velocities used to update the positions. 1 is implicit, 0 is explicit. (Warning: not relevant if use_velocity=1)"))

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

    if( use_velocity.getValue() )
    {
        f *= params.dt();  // f = h f0
        mop.addMBKv( f, 1, 0, 0 ); // f = Mv + h.f0 = p + h.f0
    }
    else  mop.addMBKv( f, -f_rayleighMass.getValue(), 0, implicitVelocity.getValue() * ( params.dt()+f_rayleighStiffness.getValue() ) ); // f = f0 + (h+rs) K v - rm M v


    //     mop.projectResponse(f); // not necessary, filtered by the projection matrix in rhs
}


void AssembledSolver::propagate(const core::MechanicalParams* params) {
	simulation::MechanicalPropagatePositionAndVelocityVisitor bob( params );
	send( bob );
}			


core::MechanicalParams AssembledSolver::mparams(const core::ExecParams& params, 
                                                double dt) const {
	core::MechanicalParams res( params );
				
    // H = (1+h*rm)M - h*B - h*(h+rs)K
    // Rayleigh damping mass factor rm is used with a negative sign because it
    // is recorded as a positive real, while its force is opposed to the velocity

    res.setMFactor( 1.0 + f_rayleighMass.getValue() * dt );
    res.setBFactor( -dt );
    res.setKFactor( -dt * ( dt + f_rayleighStiffness.getValue() ) );

	res.setDt( dt );
				
    res.setImplicitVelocity( implicitVelocity.getValue() );
    res.setImplicitPosition( implicitPosition.getValue() );

	return res;
}
			
// implicit euler
linearsolver::KKTSolver::vec AssembledSolver::rhs(const system_type& sys, bool computeForce) const {
	kkt_type::vec res = kkt_type::vec::Zero( sys.size() );

    if( !computeForce ) res.head(sys.m).setZero();
    else res.head( sys.m ) = sys.P * sys.b;
	
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

	if( warm_start.getValue() ) {
		// note: warm starting velocities is somehow equivalent to zero
		// acceleration start.

		if( use_velocity.getValue() ) {

			// velocity
			res.head( sys.m ) = sys.P * sys.v;

			// lambdas
			if( sys.n ) res.tail( sys.n ) = sys.dt * sys.lambda;
			
		} else {
			// don't warm start acceleration, as it might/will result in
			// unstable behavior
			if( sys.n ) res.tail( sys.n ) = sys.lambda;
		}
	}

	return res;	
}


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


void AssembledSolver::solve(const core::ExecParams* params,
                   double dt,
                   core::MultiVecCoordId posId,
                   core::MultiVecDerivId velId,
                   bool computeForce, // should the right part of the implicit system be computed?
                   bool integratePosition // should the position be updated?
                   )
{
    // assembly visitor
    core::MechanicalParams mparams = this->mparams(*params, dt);
    simulation::AssemblyVisitor* v = new simulation::AssemblyVisitor(&mparams, velId, lagrange.id()/*, f_rayleighStiffness.getValue(), f_rayleighMass.getValue()*/ );
    solve( params, dt, posId, velId, computeForce, integratePosition, v );
    delete v;
}


void AssembledSolver::solve(const core::ExecParams* params,
                            double dt, 
                            sofa::core::MultiVecCoordId posId,
                            sofa::core::MultiVecDerivId velId,
                            bool computeForce,
                            bool integratePosition,simulation::AssemblyVisitor * v) {
    assert(kkt);

	// obtain mparams
	core::MechanicalParams mparams = this->mparams(*params, dt);
				
	// compute forces
    if( computeForce ) forces( mparams );

     // assembly visitor (keep an accessible pointer from another component all along the solving)
     _assemblyVisitor = v;

	// fetch data
    send( *_assemblyVisitor );
	
	typedef linearsolver::AssembledSystem system_type;
	
	sofa::helper::AdvancedTimer::stepBegin( "assembly" );
    system_type sys = _assemblyVisitor->assemble();
	sofa::helper::AdvancedTimer::stepEnd( "assembly" );
	
	if( debug.getValue() ){
		sys.debug();
	}
	
	// solution vector
	system_type::vec x = warm( sys );
	

#ifdef GR_BENCHMARK
        helper::system::thread::ctime_t tfreq = helper::system::thread::CTime::getTicksPerSec();
        helper::system::thread::ctime_t t = helper::system::thread::CTime::getFastTime();
#endif


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
            assert( use_velocity.getValue() && "does not work on acceleration !");

            // stabilization
            system_type::vec tmp_b = system_type::vec::Zero(sys.size());
            tmp_b.tail( sys.n ) = b.tail(sys.n).array() * mask.array();

            system_type::vec tmp_x = system_type::vec::Zero( sys.size() );

            kkt->solve(tmp_x, sys, tmp_b);

            _assemblyVisitor->distribute_master( velId, velocity(sys, tmp_x) );

            if( integratePosition ) integrate( &mparams, posId, velId );

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
			propagate_visitor prop( &mparams );
			prop.out = core::VecId::force();
			prop.in = lagrange.id();
			
			send( prop );
		}
	}

    if( use_velocity.getValue() || implicitPosition.getValue()==1 )
    {
        // distribute (projected) velocities
        _assemblyVisitor->distribute_master( velId, velocity(sys, x) );
        // update positions
        if( integratePosition ) integrate( &mparams, posId, velId );
    }
    else // acceleration with implicitPosition!=1
    {
        // allocate a MultiVec for dv
        sofa::simulation::common::VectorOperations vop( params, this->getContext() );
        MultiVecDeriv dv(&vop, core::VecDerivId::dx() );
        // distribute dv
        _assemblyVisitor->distribute_master( dv.id(), sys.P * sys.dt * x.head( sys.m ) );
        // integrate positions and velocities (pos must be upated before vel and dv must be known)
        if( integratePosition ) integrate( &mparams, posId, velId, dv.id() );
    }


}

			
			
void AssembledSolver::init() {
				
	// let's find a KKTSolver 
	kkt = this->getContext()->get<kkt_type>();
				
	// TODO less dramatic error
	if( !kkt ) throw std::logic_error("AssembledSolver needs a KKTSolver lol");

    if( use_velocity.getValue() && ( implicitVelocity.getValue()!=1 || implicitPosition.getValue() != 1 ) )
    {
        serr<<"Using use_velocity=1, implicitVelocity & implicitPosition forced to 1"<<sendl;
        implicitVelocity.setValue(1);
        implicitPosition.setValue(1);
    }
				
}



}
}
}
