#include "CompliantImplicitSolver.h"

#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <sofa/core/ObjectFactory.h>

#include <Compliant/assembly/AssemblyVisitor.h>
#include <Compliant/utils/scoped.h>
#include <Compliant/numericalsolver/KKTSolver.h>

namespace sofa {
namespace component {
namespace odesolver {

SOFA_DECL_CLASS(CompliantImplicitSolver)
int CompliantImplicitSolverClass = core::RegisterObject("Newmark-beta implicit solver for constrained dynamics using assembly")
        .add< CompliantImplicitSolver >()
        .addAlias("AssembledSolver"); // deprecated, for backward compatibility only

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace core::behavior;





    CompliantImplicitSolver::CompliantImplicitSolver()
        : stabilization(initData(&stabilization,
                                 "stabilization",
                                 "apply a stabilization pass on kinematic constraints requesting it")),
          warm_start(initData(&warm_start,
                              true,
                              "warm_start",
                              "warm start iterative solvers: avoids biasing solution towards zero (and speeds-up resolution)")),
          debug(initData(&debug,
                         false,
                         "debug",
                         "print debug stuff")),
          constraint_forces(initData(&constraint_forces,
                                     "constraint_forces",
                                     "add constraint forces to mstate's 'force' vector at compliance levels at the end of the time step. (0->do nothing, 1->add to existing forces, 2->clear existing forces, 3-> clear existing forces and propagate constraint forces toward independent dofs) ")),
          alpha( initData(&alpha,
                          SReal(1),
                          "implicitVelocity",
                          "Weight of the next forces in the average forces used to update the velocities. 1 is implicit, 0 is explicit.")),
          beta( initData(&beta,
                         SReal(1),
                         "implicitPosition",
                         "Weight of the next velocities in the average velocities used to update the positions. 1 is implicit, 0 is explicit.")),

          stabilization_damping(initData(&stabilization_damping,
									   SReal(1e-7),
									   "stabilization_damping",
									   "stabilization damping hint to relax infeasible problems"))

          , formulation(initData(&formulation,
            "formulation",
            "solve dynamics on velocity (vel) or velocity variation (dv) or acceleration (acc)"))

          , neglecting_compliance_forces_in_geometric_stiffness(initData(&neglecting_compliance_forces_in_geometric_stiffness,
            true,
            "neglecting_compliance_forces_in_geometric_stiffness",
            "isn't the name clear enough?"))
    {
        storeDSol = false;
        assemblyVisitor = NULL;

        helper::OptionsGroup stabilizationOptions;
        stabilizationOptions.setNbItems( NB_STABILIZATION );
        stabilizationOptions.setItemName( NO_STABILIZATION,   "no stabilization"   );
        stabilizationOptions.setItemName( PRE_STABILIZATION,  "pre-stabilization"  );
        stabilizationOptions.setItemName( POST_STABILIZATION_RHS, "post-stabilization rhs" );
        stabilizationOptions.setItemName( POST_STABILIZATION_ASSEMBLY, "post-stabilization assembly" );
        stabilizationOptions.setSelectedItem( PRE_STABILIZATION );
        stabilization.setValue( stabilizationOptions );

        helper::OptionsGroup formulationOptions;
        formulationOptions.setNbItems( NB_FORMULATION );
        formulationOptions.setItemName( FORMULATION_VEL, "vel" );
        formulationOptions.setItemName( FORMULATION_DV,  "dv"  );
        formulationOptions.setItemName( FORMULATION_ACC, "acc" );
        formulationOptions.setSelectedItem( FORMULATION_VEL );
        formulation.setValue( formulationOptions );

        helper::OptionsGroup constraint_forcesOptions;
        constraint_forcesOptions.setNbItems( 4 );
        constraint_forcesOptions.setItemName( 0, "no" );
        constraint_forcesOptions.setItemName( 1, "add"  );
        constraint_forcesOptions.setItemName( 2, "clear" );
        constraint_forcesOptions.setItemName( 3, "propagate" );
        constraint_forcesOptions.setSelectedItem( 0 );
        constraint_forces.setValue( constraint_forcesOptions );
    }




    void CompliantImplicitSolver::send(simulation::Visitor& vis, bool precomputedTraversalOrder) {
//        scoped::timer step("visitor execution");

        this->getContext()->executeVisitor( &vis, precomputedTraversalOrder );

    }


    void CompliantImplicitSolver::storeDynamicsSolution(bool b) { storeDSol = b; }


    void CompliantImplicitSolver::integrate( SolverOperations& sop,
                                     const core::MultiVecCoordId& posId,
                                     const core::MultiVecDerivId& velId ) {
        scoped::timer step("position integration");
        const SReal& dt = sop.mparams().dt();

        sop.vop.v_op( posId, posId, velId, beta.getValue() * dt );
    }

    void CompliantImplicitSolver::integrate( SolverOperations& sop,
                                     const core::MultiVecCoordId& posId,
                                     const core::MultiVecDerivId& velId,
                                     const core::MultiVecDerivId& accId,
                                     const SReal& accFactor ) {
        scoped::timer step("position integration");
        const SReal& dt = sop.mparams().dt();

        core::behavior::BaseMechanicalState::VMultiOp multi;
        multi.resize(2);

        const SReal& beta = this->beta.getValue();

        multi[0].first = velId;
        multi[0].second.push_back( std::make_pair(velId, 1.0) );
        multi[0].second.push_back( std::make_pair(accId, accFactor) );

        multi[1].first = posId;
        multi[1].second.push_back( std::make_pair(posId, 1.0) );
        multi[1].second.push_back( std::make_pair(velId, dt*beta) );

        sop.vop.v_multiop( multi );
    }

    CompliantImplicitSolver::~CompliantImplicitSolver() {
        if( assemblyVisitor ) delete assemblyVisitor;
    }

    void CompliantImplicitSolver::reset() {
        if( !lagrange.id().isNull() )
        {
            // warning, vop.v_clear would only clear independent dofs
            simulation::MechanicalVOpVisitor clearVisitor(core::ExecParams::defaultInstance(), lagrange.id(), core::ConstMultiVecId::null(), core::ConstMultiVecId::null(), 1.0);
            clearVisitor.only_mapped = true;
            send( clearVisitor, false );
        }
    }

    void CompliantImplicitSolver::cleanup() {
        sofa::simulation::common::VectorOperations vop( core::ExecParams::defaultInstance(), this->getContext() );
        vop.v_free( lagrange.id(), false, true );
        vop.v_free( _ck.id(), false, true );
        vop.v_free( _acc.id() );
    }


    void CompliantImplicitSolver::compute_forces(SolverOperations& sop,
                                                  core::behavior::MultiVecDeriv& f,
                                                  core::behavior::MultiVecDeriv* f_k)
    {

        scoped::timer step("compute_forces");

        const SReal factor = ( (formulation.getValue().getSelectedId()==FORMULATION_ACC) ? 1.0 : sop.mparams().dt() );

        {
            scoped::timer substep("forces computation");

            sop.mop.computeForce( f ); // f = fk
        }

        if( f_k )
        {
            scoped::timer substep("f_k = fk");
            f_k->eq( f, factor );
        }

        if( sys.n && !neglecting_compliance_forces_in_geometric_stiffness.getValue() )
        {
            scoped::timer substep("f += fc");

            // using lambdas computed at the previous time step as an approximation of the forces generated by compliances
            // at the current configuration (due to non-linearities, after integration of the previous time-step,
            // elastic forces are too large leading to locking)
            // If no lambdas are known (e.g. first time-step, newly created constraint, contact w/o history...),
            // 0 is used and the associated compliance won't contribute to geometric stiffness generation for this step,
            // like regular constraints.
            simulation::MechanicalAddComplianceForce lvis( &sop.mparams(), f, lagrange, factor ); // f += fc  with  fc = lambda / dt
            send( lvis );
        }

        // computing mapping geometric stiffnesses based on child force stored in f
        simulation::MechanicalComputeGeometricStiffness gsvis( &sop.mparams(), f );
        send( gsvis );
    }



    void CompliantImplicitSolver::propagate(const core::MechanicalParams* params)
    {
        simulation::MechanicalProjectPositionAndVelocityVisitor project( params );
        send( project );
        simulation::MechanicalPropagateOnlyPositionAndVelocityVisitor bob( params );
        send( bob );
    }

    void CompliantImplicitSolver::filter_constraints( const core::MultiVecCoordId& posId) const {

        // compliant dofs
        for(unsigned i = 0, end = sys.compliant.size(); i < end; ++i) {
            system_type::dofs_type* dofs = sys.compliant[i];
            const system_type::constraint_type& constraint = sys.constraints[i];

            if( !constraint.projector ) continue; // if bilateral nothing to filter

            const unsigned dim = dofs->getSize(); // nb lines per constraint
            const unsigned constraint_dim = dofs->getDerivDimension(); // nb lines per constraint
            constraint.value->filterConstraints( const_cast<system_type::constraint_type*>(&constraint)->projector->mask, posId, dim, constraint_dim );
        }
    }

    void CompliantImplicitSolver::clear_constraints() const {
        // compliant dofs
        for(unsigned i = 0, end = sys.compliant.size(); i < end; ++i) {

            const system_type::constraint_type& constraint = sys.constraints[i];

            if( !constraint.projector ) continue; // if bilateral nothing to filter

            constraint.projector->mask = NULL;
            constraint.value->clear();
        }
    }


    void CompliantImplicitSolver::rhs_dynamics(SolverOperations& sop,
                                               vec& res, const system_type& sys,
                                               MultiVecDeriv& f_k,
                                               core::MultiVecCoordId posId,
                                               core::MultiVecDerivId velId ) const {
        assert( res.size() == sys.size() );


        SReal m_factor, b_factor, k_factor;
        const SReal& h = sys.dt;

        SReal alpha = this->alpha.getValue();

        switch( formulation.getValue().getSelectedId() )
        {
        case FORMULATION_DV: // c_k = h.f_k + h²Kv + hDv
            m_factor = 0.0;
            b_factor = h;
            k_factor = h*h*alpha;
            break;
        case FORMULATION_ACC: // c_k = f_k + hKv + Dv
            m_factor = 0.0;
            b_factor = 1.0;
            k_factor = h*alpha;
            break;
        case FORMULATION_VEL: // c_k = h.f_k + Mv
        default:
            // M v_k
            m_factor = 1;
            // h (1-alpha) B v_k
            b_factor = h * (1 - alpha);
            // h^2 alpha (1 - beta ) K v_k
            k_factor = h * h * alpha * (1 - beta.getValue());
            break;
        }

        {
            scoped::timer substep("c_k+=MBKv");

            // note: K v_k factor only for stiffness dofs
            sop.mop.addMBKv( f_k, m_factor, b_factor, k_factor );
        }


        unsigned off = 0;

        // master dofs: fetch what has been computed during compute_forces
        for(unsigned i = 0, end = sys.master.size(); i < end; ++i) {
            system_type::dofs_type* dofs = sys.master[i];

            unsigned dim = dofs->getMatrixSize();

            dofs->copyToBuffer(&res(off), f_k.id().getId(dofs), dim);

            off += dim;
        }


        // Applying the projection here (in flat vector representation) is great.
        // In compute_forces (in multivec representation) it would require a visitor (more expensive).
        res.head( sys.m ) = sys.P * res.head( sys.m );

        vec res_constraint(sys.n); // todo remove this temporary
        rhs_constraints_dynamics( res_constraint, sys, posId, velId );
        res.tail(sys.n).noalias() = res_constraint;
    }


    void CompliantImplicitSolver::rhs_constraints_dynamics(vec& res, const system_type& sys,
                                                   core::MultiVecCoordId posId,
                                                   core::MultiVecDerivId velId ) const {
        assert( res.size() == sys.n );

        if( !sys.n ) return;

        unsigned off = 0;

        unsigned stabilization = this->stabilization.getValue().getSelectedId();
        unsigned formulation = this->formulation.getValue().getSelectedId();

        SReal alpha = this->alpha.getValue();
        SReal beta = this->beta.getValue();


        // compliant dofs
        for(unsigned i = 0, end = sys.compliant.size(); i < end; ++i) {
            system_type::dofs_type* dofs = sys.compliant[i];
            const system_type::constraint_type& constraint = sys.constraints[i];

            const unsigned dim = dofs->getSize(); // nb constraints
            const unsigned constraint_dim = dofs->getDerivDimension(); // nb lines per constraint
            const unsigned total_dim = dim*constraint_dim; // nb constraint lines

            assert( constraint.value ); // at least a fallback must be added in filter_constraints
            constraint.value->dynamics( &res(off), dim, constraint_dim, stabilization!=NO_STABILIZATION, posId, velId );

            // adjust compliant value based on alpha/beta


            switch( formulation )
            {
            case FORMULATION_VEL: // ( phi/a -(1-b)*J.v ) / b
            {
                if( alpha != 1 ) res.segment(off,total_dim) /= alpha;
                if( beta != 1 )
                {
                    // get constraint violation velocity
                    vec v_constraint( total_dim );
                    dofs->copyToBuffer( &v_constraint(0), velId.getId(dofs), total_dim );

                    res.segment(off,total_dim).noalias() -= (1 - beta) * v_constraint;
                    res.segment(off,total_dim) /= beta;
                }
                break;
            }
            case FORMULATION_DV: // ( phi/a -J.v ) / b
            {
                if( alpha != 1 ) res.segment(off,total_dim) /= alpha;

                vec v_constraint( total_dim );
                dofs->copyToBuffer( &v_constraint(0), velId.getId(dofs), total_dim );

                res.segment(off,total_dim).noalias() -= v_constraint;

                if( beta != 1 ) res.segment(off,total_dim) /= beta;

                break;
            }
            case FORMULATION_ACC: // ( phi/a -J.v ) / bh
            {
                if( alpha != 1 ) res.segment(off,total_dim) /= alpha;

                vec v_constraint( total_dim );
                dofs->copyToBuffer( &v_constraint(0), velId.getId(dofs), total_dim );

                res.segment(off,total_dim).noalias() -= v_constraint;

                res.segment(off,total_dim) /= ( beta * sys.dt );

                break;
            }
            }

            off += total_dim;
        }
        assert( off == sys.n );



//        // adjust compliant value based on alpha/beta
//        if(alpha != 1 ) res.tail( sys.n ) /= alpha;

//        if( beta != 1 ) {
//            // TODO dofs->copyToBuffer(v_compliant, core::VecDerivId::vel(), dim); rather than sys.J * v, v_compliant is already mapped
//            // TODO use v_compliant to implement constraint damping
//            res.tail( sys.n ).noalias() -= (1 - beta) * (sys.J * v);
//            res.tail( sys.n ) /= beta;
//        }
    }

    void CompliantImplicitSolver::rhs_correction(vec& res, const system_type& sys,
                                         core::MultiVecCoordId posId,
                                         core::MultiVecDerivId velId) const {
        assert( res.size() == sys.size() );

        // master dofs
        res.head( sys.m ).setZero();
        unsigned off = sys.m;

        // compliant dofs

        for(unsigned i = 0, end = sys.compliant.size(); i < end; ++i) {
            system_type::dofs_type* dofs = sys.compliant[i];
            const system_type::constraint_type& constraint = sys.constraints[i];

            const unsigned dim = dofs->getSize(); // nb constraints
            const unsigned constraint_dim = dofs->getDerivDimension(); // nb lines per constraint

            assert( constraint.value ); // at least a fallback must be added in filter_constraints

            constraint.value->correction( &res(off), dim, constraint_dim, posId, velId );

            off += dim * constraint_dim;
        }
    }


    void CompliantImplicitSolver::get_state(vec& res, const system_type& sys, const core::MultiVecDerivId& multiVecId) const {

        assert( res.size() == sys.size() );

        unsigned off = 0;

        for(unsigned i = 0, end = sys.master.size(); i < end; ++i) {
            system_type::dofs_type* dofs = sys.master[i];

            unsigned dim = dofs->getMatrixSize();

            dofs->copyToBuffer(&res(off), multiVecId.getId(dofs), dim);
            off += dim;
        }


        for(unsigned i = 0, end = sys.compliant.size(); i < end; ++i) {
            system_type::dofs_type* dofs = sys.compliant[i];

            unsigned dim = dofs->getMatrixSize();

            dofs->copyToBuffer(&res(off), lagrange.id().getId( dofs ), dim);

            off += dim;
        }

        assert( off == sys.size() );

    }


    void CompliantImplicitSolver::set_state_v(const system_type& sys, const vec& data, const core::MultiVecDerivId& velId) const {

        assert( data.size() == sys.size() );

        unsigned off = 0;

        // TODO project v ?
        for(unsigned i = 0, end = sys.master.size(); i < end; ++i) {
            system_type::dofs_type* dofs = sys.master[i];

            unsigned dim = dofs->getMatrixSize();

            dofs->copyFromBuffer(velId.getId(dofs), &data(off), dim);
            off += dim;
        }
    }


    void CompliantImplicitSolver::set_state_lambda(const system_type& sys, const vec& data) const {

        // we directly store lambda (and not constraint *force*)
        unsigned off = 0;

        for(unsigned i = 0, end = sys.compliant.size(); i < end; ++i) {
            system_type::dofs_type* dofs = sys.compliant[i];

            unsigned dim = dofs->getMatrixSize();

            dofs->copyFromBuffer(lagrange.id().getId( dofs ), &data(off), dim);

            off += dim;
        }
    }

    void CompliantImplicitSolver::set_state(const system_type& sys, const vec& data, const core::MultiVecDerivId& velId) const {
        set_state_v( sys, data, velId );
        set_state_lambda( sys, data.tail(sys.n) );
    }



    void CompliantImplicitSolver::perform_assembly( const core::MechanicalParams *mparams, system_type& sys )
    {
        scoped::timer step("perform_assembly");

        // max: il ya des smart ptr pour ca.
        if( assemblyVisitor ) delete assemblyVisitor;
        assemblyVisitor = new simulation::AssemblyVisitor(mparams);

        // fetch nodes/data
        {
            scoped::timer step("assembly: fetch data");
            send( *assemblyVisitor );
        }

        // assemble system
        assemblyVisitor->assemble(sys);
    }

    void CompliantImplicitSolver::solve(const core::ExecParams* params,
                                SReal dt,
                                core::MultiVecCoordId posId,
                                core::MultiVecDerivId velId) {

        static_cast<simulation::Node*>(getContext())->precomputeTraversalOrder( params );

        assert(kkt);

        bool useVelocity = (formulation.getValue().getSelectedId()==FORMULATION_VEL);

        SolverOperations sop( params, this->getContext(), alpha.getValue(), beta.getValue(), dt, posId, velId, true );


        MultiVecDeriv f( &sop.vop, core::VecDerivId::force() ); // total force (stiffness + compliance) (f_k term)
        _ck.realloc( &sop.vop, false, true ); // the right part of the implicit system (c_k term)

        if( !useVelocity ) _acc.realloc( &sop.vop );

        {
            scoped::timer step("lambdas alloc");
            lagrange.realloc( &sop.vop, false, true );
//            vop.print( lagrange,serr, "BEGIN lambda: ", "\n" );
        }

        // compute forces and implicit right part warning: must be
        // call before assemblyVisitor since the mapping's geometric
        // stiffness depends on its child force
        compute_forces( sop, f, &_ck );

        // assemble system
        perform_assembly( &sop.mparams(), sys );

        // debugging
        if( debug.getValue() ) sys.debug();
        if( f_printLog.getValue() )
        {
            sout << "dynamics size m: " <<sys.m<< sendl;
            sout << "constraint size n: " <<sys.n<< sendl;
        }

        // look for violated and active constraints
        // must be performed after assembly and before system factorization
        filter_constraints( posId );

        // system factor
        {
            scoped::timer step("system factor");
            kkt->factor( sys );
        }

        // system solution / rhs
        vec x(sys.size());
        vec rhs(sys.size());

        // ready to solve
        {

            unsigned stabilizationType = stabilization.getValue().getSelectedId();

            scoped::timer step("system solve");

            // constraint pre-stabilization
            if( sys.n && stabilizationType==PRE_STABILIZATION )
            {
                // Note that stabilization is always solved in velocity

                scoped::timer step("correction");

                x = vec::Zero( sys.size() );
                rhs_correction(rhs, sys, posId, velId);

                kkt->correct(x, sys, rhs, stabilization_damping.getValue() );

                if( debug.getValue() ) {
                    serr << "correction rhs:" << std::endl
                              << rhs.transpose() << std::endl
                              << "solution:" << std::endl
                              << x.transpose() << sendl;
                }

                MultiVecDeriv v_stab( &sop.vop ); // a temporary multivec not to modify velocity
                set_state_v( sys, x, v_stab.id() ); // set v (no need to set lambda)
                integrate( sop, posId, v_stab.id() );
            }

            // actual dynamics
            {
                scoped::timer step("dynamics");

                if( warm_start.getValue() ) get_state( x, sys, useVelocity ? velId : _acc.id() );
                else x = vec::Zero( sys.size() );

                rhs_dynamics( sop, rhs, sys, _ck, posId, velId );

                kkt->solve(x, sys, rhs);
				
                if( debug.getValue() ) {
                    sout << "dynamics rhs:" << std::endl
                              << rhs.transpose() << std::endl
                              << "solution:" << std::endl
                              << x.transpose() << sendl;
                }
                if( storeDSol ) {
                    dynamics_rhs = rhs;
                    dynamics_solution = x;
                }


                switch( formulation.getValue().getSelectedId() )
                {
                case FORMULATION_VEL: // p+ = p- + h.v
                    set_state( sys, x, velId ); // set v and lambda
                    integrate( sop, posId, velId );
                    break;
                case FORMULATION_DV: // v+ = v- + dv     p+ = p- + h.v
                    set_state( sys, x, _acc.id() ); // set v and lambda
                    integrate( sop, posId, velId, _acc.id(), 1.0  );
                    break;
                case FORMULATION_ACC: // v+ = v- + h.a   p+ = p- + h.v
                    set_state( sys, x, _acc.id() ); // set v and lambda
                    integrate( sop, posId, velId, _acc.id(), dt  );
                    break;
                }

                // TODO is this even needed at this point ? NO it is done by the animation loop
//                propagate( &sop.mparams() );
            }


            // propagate lambdas if asked to (constraint forces must be propagated before post_stab)
            unsigned constraint_f = constraint_forces.getValue().getSelectedId();
            if( constraint_f && sys.n ) {
                scoped::timer step("constraint_forces");
                simulation::propagate_constraint_force_visitor prop( &sop.mparams(), core::VecId::force(), lagrange.id(), useVelocity ? 1.0/sys.dt : 1.0, constraint_f>1, constraint_f==3 );
                send( prop );
            }

            // constraint post-stabilization
            if( stabilizationType>=POST_STABILIZATION_RHS )
            {
                post_stabilization( sop, posId, velId, stabilizationType==POST_STABILIZATION_ASSEMBLY);
            }
        }


        clear_constraints();


//        vop.print( lagrange,serr, "END lambda: ", "\n" );


    }




    void CompliantImplicitSolver::init() {

        // do want KKTSolver !
        kkt = this->getContext()->get<kkt_type>(core::objectmodel::BaseContext::Local);

        // TODO slightly less dramatic error, maybe ?
        if( !kkt ) throw std::logic_error("CompliantImplicitSolver needs a KKTSolver");
    }





    void CompliantImplicitSolver::post_stabilization(SolverOperations& sop,
                                              core::MultiVecCoordId posId, core::MultiVecDerivId velId,
                                              bool fullAssembly, bool realloc )
    {
        if( !sys.n ) return;

        if( realloc )
        {
            static_cast<simulation::Node*>(getContext())->precomputeTraversalOrder( &sop.mparams() ); // if the graph changed, the traversal order needs to be updated
            _ck.realloc( &sop.vop, false, true ); // the right part of the implicit system (c_k term)
        }

        // Note that stabilization is always solved in velocity

        scoped::timer step("correction");

        // propagate dynamics position and velocity
        propagate( &sop.mparams() );

        // at this point collision detection should be run again
        // for now, we keep the same contact points with the same normals (wrong)
        // but the contact violation is updated
        // dt must be small with collisions

        MultiVecDeriv f( &sop.vop, core::VecDerivId::force() );

        compute_forces( sop, f );

        if( fullAssembly )
        {
            // assemble system
            perform_assembly( &sop.mparams(), sys );

            // look for violated and active constraints
            // must be performed after assembly and before system factorization
            filter_constraints( posId );

            // system factor
            {
                scoped::timer step("correction system factor");
                kkt->factor( sys );
            }
        }
        else
        {
            filter_constraints( posId );
        }

        vec x = vec::Zero( sys.size() );
        vec rhs( sys.size() );
        rhs_correction(rhs, sys, posId, velId);

        kkt->correct(x, sys, rhs, stabilization_damping.getValue() );

        if( debug.getValue() ) {
            sout << "correction rhs:" << std::endl
                      << rhs.transpose() << std::endl
                      << "solution:" << std::endl
                      << x.transpose() << sendl;
        }

        MultiVecDeriv v_stab( &sop.vop );  // a temporary multivec not to modify velocity
        set_state_v( sys, x, v_stab.id() ); // set v (no need to set lambda)
        integrate( sop, posId, v_stab.id() );

        propagate( &sop.mparams() );
    }


    void CompliantImplicitSolver::parse(core::objectmodel::BaseObjectDescription* arg)
    {
        Inherit1::parse(arg);

        // to be backward compatible with previous data structure
        const char* propagate_lambdasChar = arg->getAttribute("propagate_lambdas");
        if( propagate_lambdasChar )
        {
            serr<<helper::logging::Message::Deprecated<<"parse: You are using a deprecated Data 'propagate_lambdas', please have a look to 'constraint_forces'"<<sendl;
            Data<bool> data;
            data.read( propagate_lambdasChar );
            if( data.getValue() )
                constraint_forces.beginWriteOnly()->setSelectedItem(3); constraint_forces.endEdit();
        }
    }

}
}
}
