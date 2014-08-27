#include "CompliantImplicitSolver.h"

#include <sofa/component/linearsolver/EigenSparseMatrix.h>
#include <sofa/component/linearsolver/EigenVector.h>
#include <sofa/core/ObjectFactory.h>

#include "assembly/AssemblyVisitor.h"
#include "utils/scoped.h"

using std::cerr;
using std::endl;

namespace sofa {
namespace component {
namespace odesolver {

SOFA_DECL_CLASS(CompliantImplicitSolver);
int CompliantImplicitSolverClass = core::RegisterObject("Example compliance solver using assembly")
        .add< CompliantImplicitSolver >()
        .addAlias("AssembledSolver"); // deprecated, for backward compatibility only

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace core::behavior;




    propagate_constraint_force_visitor::propagate_constraint_force_visitor(const sofa::core::MechanicalParams* mparams,
                                         const core::MultiVecDerivId& force,
                                         const core::MultiVecDerivId& lambda,
                                         SReal dt)
        : simulation::MechanicalVisitor(mparams)
        , force( force )
        , lambda( lambda )
        , invdt( 1.0/dt )
    {}

    propagate_constraint_force_visitor::Result propagate_constraint_force_visitor::fwdMechanicalState(simulation::Node* /*node*/,
                                                                    core::behavior::BaseMechanicalState* state) {
        // compliance cannot be present at independent dof level
        state->resetForce(mparams, force.getId(state));
        return RESULT_CONTINUE;
    }

    propagate_constraint_force_visitor::Result propagate_constraint_force_visitor::fwdMappedMechanicalState(simulation::Node* node,
                                                                          core::behavior::BaseMechanicalState* state) {

        // lambdas should only be present at compliance location

        if( !node->forceField.empty() && node->forceField[0]->isCompliance.getValue() )
            // compliance should be alone in the node
            state->vOp( mparams, force.getId(state), core::ConstVecId::null(), lambda.getId(state), invdt ); // constraint force = lambda / dt
        else
            state->resetForce(mparams, force.getId(state));

        return RESULT_CONTINUE;
    }


    void propagate_constraint_force_visitor::bwdMechanicalMapping(simulation::Node* /*node*/,
                                                        core::BaseMapping* map) {
        map->applyJT(mparams, force, force);
    }

    void propagate_constraint_force_visitor::bwdProjectiveConstraintSet(simulation::Node* /*node*/,
                                                              core::behavior::BaseProjectiveConstraintSet* c) {
        c->projectResponse( mparams, force );
    }






    CompliantImplicitSolver::CompliantImplicitSolver()
        : stabilization(initData(&stabilization,
                                 "stabilization",
                                 "apply a stabilization pass on kinematic constraints requesting it")),
          warm_start(initData(&warm_start,
                              true,
                              "warm_start",
                              "warm start iterative solvers: avoids biasing solution towards zero (and speeds-up resolution)")),
          propagate_lambdas(initData(&propagate_lambdas,
                                     false,
                                     "propagate_lambdas",
                                     "propagate Lagrange multipliers in force vector at the end of time step")),
          debug(initData(&debug,
                         false,
                         "debug",
                         "print debug stuff")),
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
    }




    void CompliantImplicitSolver::send(simulation::Visitor& vis) {
//        scoped::timer step("visitor execution");

        this->getContext()->executeVisitor( &vis );

    }


    void CompliantImplicitSolver::storeDynamicsSolution(bool b) { storeDSol = b; }


    void CompliantImplicitSolver::integrate( const core::MechanicalParams* params,
                                     const core::MultiVecCoordId& posId,
                                     const core::MultiVecDerivId& velId ) {
        scoped::timer step("position integration");
        SReal dt = params->dt();

        // integrate positions
        sofa::simulation::common::VectorOperations vop( params, this->getContext() );

        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp multi;

        multi.resize(1);

        multi[0].first = posId;
        multi[0].second.push_back( std::make_pair(posId, 1.0) );
        multi[0].second.push_back( std::make_pair(velId, beta.getValue() * dt) );

        vop.v_multiop( multi );
    }



    CompliantImplicitSolver::~CompliantImplicitSolver() {
        if( assemblyVisitor ) delete assemblyVisitor;
    }

    void CompliantImplicitSolver::cleanup() {
        sofa::simulation::common::VectorOperations vop( core::ExecParams::defaultInstance(), this->getContext() );
        vop.v_free( lagrange.id(), false, true );
        vop.v_free( _ck.id(), false, true );
    }


    // this is c_k computation (see compliant-reference.pdf, section 3)
    // at the end, the multivec f MUST contain the forces
    // (to compute mapping's geometric stiffnesses during assembly)
    void CompliantImplicitSolver::compute_forces(SolverOperations& sop,
                                         core::behavior::MultiVecDeriv& f,
                                         core::behavior::MultiVecDeriv& c )
    {

        scoped::timer step("implicit rhs computation");


        const SReal h = sop.mparams().dt();


        {
            scoped::timer substep("forces computation");

            sop.mop.computeForce( f ); // f = fk
        }

        {
            scoped::timer substep("c_k = h.fk");

            c.eq( f, h );
        }

        if( !neglecting_compliance_forces_in_geometric_stiffness.getValue() )
        {
            scoped::timer substep("f += fc");

            // using lambdas computed at the previous time step as an approximation of the forces generated by compliances
            // If no approximation are known, 0 is used and the associated compliance won't contribute
            // to geometric sitffness generation for this step.
            simulation::MechanicalAddComplianceForce lvis( &sop.mparams(), f, lagrange, h ); // f += fc   f += lambda / dt
            send( lvis );

            // TODO have a look about reseting or not forces of mapped dofs
        }


        {
            scoped::timer substep("c_k+=MBKv");

            // M v_k
            const SReal mfactor = 1;

            // h (1-alpha) B v_k
            const SReal bfactor = h * (1 - alpha.getValue());

            // h^2 alpha (1 - beta ) K v_k
            const SReal kfactor = h * h * alpha.getValue() * (1 - beta.getValue());

            // note: K v_k factor only for stiffness dofs
            sop.mop.addMBKv( c, mfactor, bfactor, kfactor );
        }

    }



    void CompliantImplicitSolver::propagate(const core::MechanicalParams* params)
    {
        simulation::MechanicalPropagatePositionAndVelocityVisitor bob( params );
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
            constraint.value->filterConstraints( constraint.projector->mask, posId, dim, constraint_dim );
        }
    }

    void CompliantImplicitSolver::clear_constraints() const {
        // compliant dofs
        for(unsigned i = 0, end = sys.compliant.size(); i < end; ++i) {

            const system_type::constraint_type& constraint = sys.constraints[i];

            if( !constraint.projector ) continue; // if bilateral nothing to filter

            constraint.projector->mask.clear();
            constraint.value->clear();
        }
    }


    void CompliantImplicitSolver::rhs_dynamics(vec& res, const system_type& sys, const MultiVecDeriv& b,
                                       core::MultiVecCoordId posId,
                                       core::MultiVecDerivId velId) const {
        assert( res.size() == sys.size() );

        unsigned off = 0;

        // master dofs: fetch what has been computed during compute_forces
        for(unsigned i = 0, end = sys.master.size(); i < end; ++i) {
            system_type::dofs_type* dofs = sys.master[i];

            unsigned dim = dofs->getMatrixSize();

            dofs->copyToBuffer(&res(off), b.id().getId(dofs), dim);

            off += dim;
        }


        // Applying the projection here (in flat vector representation) is great.
        // In compute_forces (in multivec representation) it would require a visitor (more expensive).
        res.head( sys.m ) = sys.P * res.head( sys.m );

        vec res_constraint(sys.n);
        rhs_constraints_dynamics( res_constraint, sys, posId, velId );
        res.tail(sys.n).noalias() = res_constraint;
    }


    void CompliantImplicitSolver::rhs_constraints_dynamics(vec& res, const system_type& sys,
                                                   core::MultiVecCoordId posId,
                                                   core::MultiVecDerivId velId) const {
        assert( res.size() == sys.n );

        if( !sys.n  ) return;

        unsigned off = 0;

        // compliant dofs
        for(unsigned i = 0, end = sys.compliant.size(); i < end; ++i) {
            system_type::dofs_type* dofs = sys.compliant[i];
            const system_type::constraint_type& constraint = sys.constraints[i];

            const unsigned dim = dofs->getSize(); // nb constraints
            const unsigned constraint_dim = dofs->getDerivDimension(); // nb lines per constraint

            // get constraint violation velocity
            vec v_constraint( dim*constraint_dim );
            dofs->copyToBuffer( &v_constraint(0), velId.getId(dofs), dim*constraint_dim );

            assert( constraint.value ); // at least a fallback must be added in filter_constraints

            constraint.value->dynamics( &res(off), dim, constraint_dim, stabilization.getValue().getSelectedId(), posId, velId );

            // adjust compliant value based on alpha/beta   ( phi/a -(1-b)*J.v ) / b
            if( alpha.getValue() != 1 ) res(off) /= alpha.getValue();
            if( beta.getValue() != 1 ) {
                res(off) -= (1 - beta.getValue()) * v_constraint(off);
                res(off) /= beta.getValue();
            }

            off += dim * constraint_dim;
        }
        assert( off == sys.n );



//        // adjust compliant value based on alpha/beta
//        if(alpha.getValue() != 1 ) res.tail( sys.n ) /= alpha.getValue();

//        if( beta.getValue() != 1 ) {
//            // TODO dofs->copyToBuffer(v_compliant, core::VecDerivId::vel(), dim); rather than sys.J * v, v_compliant is already mapped
//            // TODO use v_compliant to implement constraint damping
//            res.tail( sys.n ).noalias() = res.tail( sys.n ) - (1 - beta.getValue()) * (sys.J * v);
//            res.tail( sys.n ) /= beta.getValue();
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
        // max: il ya des auto_ptr pour ca.
        if( assemblyVisitor ) delete assemblyVisitor;
        assemblyVisitor = new simulation::AssemblyVisitor(mparams);

        // fetch nodes/data
        send( *assemblyVisitor );

        // assemble system
        sys = assemblyVisitor->assemble();
    }

    void CompliantImplicitSolver::solve(const core::ExecParams* params,
                                double dt,
                                core::MultiVecCoordId posId,
                                core::MultiVecDerivId velId) {
        assert(kkt);

        // mechanical parameters
//        core::MechanicalParams mparams;
//        this->buildMparams( mparams, *params, dt );
//        mparams.setX(posId);
//        mparams.setV(velId);

//        simulation::common::MechanicalOperations mop( params, this->getContext() );
//        simulation::common::VectorOperations vop( params, this->getContext() );

        SolverOperations sop( params, this->getContext(), alpha.getValue(), beta.getValue(), dt, posId, velId );


        MultiVecDeriv f( &sop.vop, core::VecDerivId::force() ); // total force (stiffness + compliance) (f_k term)
        _ck.realloc( &sop.vop, false, true ); // the right part of the implicit system (c_k term)

        {
            scoped::timer step("lambdas alloc");
            lagrange.realloc( &sop.vop, false, true );
//            vop.print( lagrange,std::cerr, "BEGIN lambda: ", "\n" );
        }

        // compute forces and implicit right part warning: must be
        // call before assemblyVisitor since the mapping's geometric
        // stiffness depends on its child force
        compute_forces( sop, f, _ck );

        // assemble system
        perform_assembly( &sop.mparams(), sys );

        // debugging
        if( debug.getValue() ) sys.debug();

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
            if( sys.n && stabilizationType==PRE_STABILIZATION)
            {
                    scoped::timer step("correction");

                    x = vec::Zero( sys.size() );
                    rhs_correction(rhs, sys, posId, velId);

                    kkt->correct(x, sys, rhs, stabilization_damping.getValue() );

                    if( debug.getValue() ) {
                        std::cerr << "correction rhs:" << std::endl
                                  << rhs.transpose() << std::endl
                                  << "solution:" << std::endl
                                  << x.transpose() << std::endl;
                    }


                    MultiVecDeriv v_stab( &sop.vop ); // a temporary multivec not to modify velocity
                    set_state_v( sys, x, v_stab.id() ); // set v (no need to set lambda)
                    integrate( &sop.mparams(), posId, v_stab.id() );
            }

            // actual dynamics
            {
                scoped::timer step("dynamics");

                if( warm_start.getValue() ) /*x = current;*/  get_state( x, sys, velId );
                else x = vec::Zero( sys.size() );

                rhs_dynamics(rhs, sys, _ck, posId, velId );

                kkt->solve(x, sys, rhs);
				
                if( debug.getValue() ) {
                    std::cerr << "dynamics rhs:" << std::endl
                              << rhs.transpose() << std::endl
                              << "solution:" << std::endl
                              << x.transpose() << std::endl;
                }
                if( storeDSol ) {
                    dynamics_rhs = rhs;
                    dynamics_solution = x;
                }


                set_state( sys, x, velId ); // set v and lambda
                integrate( &sop.mparams(), posId, velId );

                // TODO is this even needed at this point ?
                propagate( &sop.mparams() );
            }


            // constraint post-stabilization
            if( stabilizationType>=POST_STABILIZATION_RHS )
            {
                post_stabilization( sop, posId, velId, stabilizationType==POST_STABILIZATION_ASSEMBLY);
            }
        }

        // propagate lambdas if asked to
        if( propagate_lambdas.getValue() && sys.n ) {
            scoped::timer step("lambda propagation");
            propagate_constraint_force_visitor prop( &sop.mparams(), core::VecId::force(), lagrange.id(), sys.dt );
            send( prop );
        }

        clear_constraints();


//        vop.print( lagrange,std::cerr, "END lambda: ", "\n" );


    }




    void CompliantImplicitSolver::init() {

        // do want KKTSolver !
        kkt = this->getContext()->get<kkt_type>(core::objectmodel::BaseContext::Local);

        // TODO slightly less dramatic error, maybe ?
        if( !kkt ) throw std::logic_error("CompliantImplicitSolver needs a KKTSolver");
    }





    void CompliantImplicitSolver::post_stabilization( SolverOperations& sop,
                                              core::MultiVecCoordId posId, core::MultiVecDerivId velId,
                                              bool fullAssembly )
    {
        if( !sys.n ) return;

        scoped::timer step("correction");

        // at this point collision detection should be run again
        // for now, we keep the same contact points with the same normals (wrong)
        // but the contact violation is updated
        // dt must be small with collisions

        vec x(sys.size()), rhs(sys.size());
        MultiVecDeriv f( &sop.vop, core::VecDerivId::force() );

        compute_forces( sop, f, _ck );

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

        x = vec::Zero( sys.size() );
        rhs_correction(rhs, sys, posId, velId);

        kkt->correct(x, sys, rhs, stabilization_damping.getValue() );

        if( debug.getValue() ) {
            std::cerr << "correction rhs:" << std::endl
                      << rhs.transpose() << std::endl
                      << "solution:" << std::endl
                      << x.transpose() << std::endl;
        }

        MultiVecDeriv v_stab( &sop.vop );  // a temporary multivec not to modify velocity
        set_state_v( sys, x, v_stab.id() ); // set v (no need to set lambda)
        integrate( &sop.mparams(), posId, v_stab.id() );

        propagate( &sop.mparams() );
    }



}
}
}
