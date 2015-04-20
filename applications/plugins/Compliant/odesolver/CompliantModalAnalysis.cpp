#include "CompliantModalAnalysis.h"

#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <SofaEigen2Solver/EigenVector.h>
#include <sofa/core/ObjectFactory.h>

#include "assembly/AssemblyVisitor.h"
#include "utils/scoped.h"

using std::cerr;
using std::endl;

namespace sofa {
namespace component {
namespace odesolver {

SOFA_DECL_CLASS(CompliantModalAnalysis)
int CompliantModalAnalysisClass = core::RegisterObject("Modal analysis")
        .add< CompliantModalAnalysis >();

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace core::behavior;





CompliantModalAnalysis::CompliantModalAnalysis()
    : CompliantImplicitSolver()
{ }




    CompliantModalAnalysis::~CompliantModalAnalysis() {
    }

    void CompliantModalAnalysis::solve(const core::ExecParams* params,
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
//            vop.print( lagrange,std::cerr, "BEGIN lambda: ", "\n" );
        }

        // compute forces and implicit right part warning: must be
        // call before assemblyVisitor since the mapping's geometric
        // stiffness depends on its child force
        compute_forces( sop, f, &_ck );

        // assemble system
        perform_assembly( &sop.mparams(), sys );

        {
        core::MechanicalParams mparams=*core::MechanicalParams::defaultInstance();
        mparams.setMFactor( 0 );
        mparams.setBFactor( 0 );
        mparams.setKFactor( 1 );
        simulation::AssemblyVisitor v(&mparams);
        this->getContext()->executeVisitor(&v);
        component::linearsolver::AssembledSystem sys;
        v.assemble(sys); // assemble system
        system_type::rmat K = sys.H;
        std::cerr << "Stiffness: " << K << std::endl;
        }

        {
        core::MechanicalParams mparams=*core::MechanicalParams::defaultInstance();
        mparams.setMFactor( 1 );
        mparams.setBFactor( 0 );
        mparams.setKFactor( 0 );
        simulation::AssemblyVisitor v(&mparams);
        this->getContext()->executeVisitor(&v);
        component::linearsolver::AssembledSystem sys;
        v.assemble(sys); // assemble system
        std::cerr << "Mass: " << sys.H << std::endl;
        }

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
                    std::cerr << "correction rhs:" << std::endl
                              << rhs.transpose() << std::endl
                              << "solution:" << std::endl
                              << x.transpose() << std::endl;
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
                    std::cerr << "dynamics rhs:" << std::endl
                              << rhs.transpose() << std::endl
                              << "solution:" << std::endl
                              << x.transpose() << std::endl;
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
            if( propagate_lambdas.getValue() && sys.n ) {
                scoped::timer step("lambda propagation");
                propagate_constraint_force_visitor prop( &sop.mparams(), core::VecId::force(), lagrange.id(), useVelocity ? sys.dt : 1.0 );
                send( prop );
            }

            // constraint post-stabilization
            if( stabilizationType>=POST_STABILIZATION_RHS )
            {
                post_stabilization( sop, posId, velId, stabilizationType==POST_STABILIZATION_ASSEMBLY);
            }
        }


        clear_constraints();


//        vop.print( lagrange,std::cerr, "END lambda: ", "\n" );


    }





}
}
}
