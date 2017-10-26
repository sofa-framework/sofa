#include "CompliantNLImplicitSolver.h"

#include <sofa/core/ObjectFactory.h>

#include <Compliant/assembly/AssemblyVisitor.h>
#include <Compliant/utils/scoped.h>
#include <Compliant/numericalsolver/KKTSolver.h>

using std::cerr;
using std::endl;

namespace sofa {
namespace component {
namespace odesolver {

SOFA_DECL_CLASS(CompliantNLImplicitSolver)
int CompliantNLImplicitSolverClass = core::RegisterObject("Implicit solver with pre-inversed matrix and Newton iterations")
        .add< CompliantNLImplicitSolver >()
        .addAlias("CompliantNonLinearImplicitSolver")
        .addAlias("NewtonSolver") // deprecated, for backward compatibility only
        ;

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace core::behavior;




CompliantNLImplicitSolver::CompliantNLImplicitSolver()
    : precision(initData(&precision,
                   (SReal) 1.0e-6,
                   "precision",
                   "stop criterion of the Newton iteration: square norm of the residual"))
    , relative(initData(&relative, false, "relative", "use relative precision") )
    , iterations(initData(&iterations,
                     (unsigned) 10,
                     "iterations",
                     "maximum number of Newton iterations"))
    , newtonStepLength(initData(&newtonStepLength,
                   (SReal) 1,
                   "newtonStepLength",
                   "amount of correction applied at each Newton iteration (a line-search is performed for newtonStepLength=1)"))
//    , staticSolver(initData(&staticSolver,
//                   false,
//                   "static",
//                   "static solver? (solving dynamics by default)  WIP not yet working"))
{
    neglecting_compliance_forces_in_geometric_stiffness.setValue( false ); // geometric stiffness generated from compliance force is important for a non-linear solver
}



void CompliantNLImplicitSolver::cleanup() {
    CompliantImplicitSolver::cleanup();

    // free temporary multivecs
    sofa::simulation::common::VectorOperations vop( core::ExecParams::defaultInstance(), this->getContext() );
    vop.v_free( _x0.id() );
    vop.v_free( _v0.id(), false, true );
    vop.v_free( _f0.id() );
    vop.v_free( _vfirstguess.id() );
    vop.v_free( _lambdafirstguess.id(), false, true );
    vop.v_free( _err.id(), false, true );
    vop.v_free( _deltaV.id(), false, true );
}




class AccumulateConstraintForceVisitor : public simulation::MechanicalVisitor
{
public:
    core::MultiVecDerivId lambda;

    AccumulateConstraintForceVisitor(const sofa::core::MechanicalParams* mparams, core::MultiVecDerivId lambda )
        : simulation::MechanicalVisitor(mparams), lambda(lambda)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual Result fwdMappedMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm)
    {
        if( node->forceField.empty() || node->forceField[0]->isCompliance.getValue() )
            mm->resetForce(this->params, lambda.getId(mm));
        return RESULT_CONTINUE;
    }


    void bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
    {
        //       cerr<<"MechanicalComputeForceVisitor::bwdMechanicalMapping "<<map->getName()<<endl;

            ForceMaskActivate(map->getMechFrom() );
            ForceMaskActivate(map->getMechTo() );

            //map->accumulateForce();
            map->applyJT(mparams, lambda, lambda);

            ForceMaskDeactivate( map->getMechTo() );

    }


    void bwdMechanicalState(simulation::Node* , core::behavior::BaseMechanicalState* mm)
    {
        mm->forceMask.activate(false);
    }


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const {return "AccumulateConstraintForceVisitor";}
    virtual std::string getInfos() const
    {
        std::string name=std::string("[")+lambda.getName()+std::string("]");
        return name;
    }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addWriteVector(lambda);
    }
#endif
};

class MechanicalMyComputeForceVisitor : public simulation::MechanicalComputeForceVisitor
{
    core::MultiVecDerivId lambdas;
     SReal invdt;
public:
    MechanicalMyComputeForceVisitor(const sofa::core::MechanicalParams* mparams, core::MultiVecDerivId res, core::MultiVecDerivId lambdas, SReal dt )
        : simulation::MechanicalComputeForceVisitor(mparams,res,true,true)
        , lambdas( lambdas )
        , invdt( -1.0/dt )
    {
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        mm->resetForce(this->params, res.getId(mm));
        mm->accumulateForce(this->params, res.getId(mm));
        return RESULT_CONTINUE;
    }
    virtual Result fwdMappedMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm)
    {
        if( !node->forceField.empty() && node->forceField[0]->isCompliance.getValue() )
            // compliance should be alone in the node
            mm->vOp( mparams, res.getId(mm), core::ConstVecId::null(), lambdas.getId(mm), invdt ); // constraint force = lambda / dt
        else
            mm->resetForce(mparams, res.getId(mm));

        mm->accumulateForce(this->params, res.getId(mm));
        return RESULT_CONTINUE;
    }

};




/// res += constraint forces (== lambda/dt), only for mechanical object linked to a compliance
class MechanicalAddLagrangeForce : public simulation::MechanicalVisitor
{
    core::MultiVecDerivId res, lambdas;
    SReal invdt;


public:
    MechanicalAddLagrangeForce(const sofa::core::MechanicalParams* mparams, core::MultiVecDerivId res, core::MultiVecDerivId lambdas, SReal dt )
        : MechanicalVisitor(mparams), res(res), lambdas(lambdas), invdt(-1.0/dt)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    // TODO how to propagate lambdas without invalidating forces on mapped dofs?
    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        mm->resetForce(this->params, res.getId(mm));
        return RESULT_CONTINUE;
    }


    virtual Result fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* ff)
    {
        if( ff->isCompliance.getValue() )
        {
            core::behavior::BaseMechanicalState* mm = ff->getContext()->getMechanicalState();
            mm->vOp( this->params, res.getId(mm), res.getId(mm), lambdas.getId(mm), invdt );
            // lambdas should always be allocated through realloc (null values for new constraints)
        }

        return RESULT_CONTINUE;
    }


    virtual void bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
    {
        ForceMaskActivate( map->getMechFrom() );
        ForceMaskActivate( map->getMechTo() );
        map->applyJT( this->mparams, res, res );
        ForceMaskDeactivate( map->getMechTo() );
    }

    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        mm->forceMask.activate(false);
    }

    virtual void bwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c)
    {
        c->projectResponse( this->mparams, res );
    }


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const {return "MechanicalAddLagrangeForce";}
    virtual std::string getInfos() const
    {
        std::string name=std::string("[")+res.getName()+","+lambdas.getName()+std::string("]");
        return name;
    }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addWriteVector(res);
        addWriteVector(lambdas);
    }
#endif
};


SReal CompliantNLImplicitSolver::compute_residual( SolverOperations sop, core::MultiVecDerivId residual,
                                      core::MultiVecCoordId newX, const core::MultiVecDerivId newV, core::MultiVecDerivId newF,
                                      core::MultiVecCoordId oldX, core::MultiVecDerivId oldV, core::MultiVecDerivId oldF,
                                      const vec& lambda, chuck_type* residual_constraints )
{
    const SReal h = sop.mparams().dt();
    const SReal a = alpha.getValue();
    const SReal b = beta.getValue();

    typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
    VMultiOp force_ops(1);
    VMultiOp velocity_ops(1);

    // average_f = (1-a).oldF + a.newF
    force_ops[0].first = residual;

    if( a!=0 )
    {

        // compute next position, based on next velocity
        // newX = oldX + h*beta*newV + h*(1-beta)*oldV
        VMultiOp opsnewX(1);
        opsnewX[0].first = newX;
        opsnewX[0].second.push_back(std::make_pair(oldX,1.0));
        if( b!=1 ) opsnewX[0].second.push_back(std::make_pair(oldV,h*(1-b) ));
        if( b!=0 ) opsnewX[0].second.push_back(std::make_pair(newV,h*b ));
        sop.vop.v_multiop(opsnewX);






        // compute the corresponding force
        sop.mop.computeForce( getContext()->getTime()+h, newF, newX, newV );

        // add eventual constraint forces
        if( sys.n )
        {
            MechanicalAddLagrangeForce lvis( &sop.mparams(), newF, lagrange, h ); // f += fc   f += lambda / dt
            send( lvis );
        }


         force_ops[0].second.push_back(std::make_pair( newF, a ));
    }

    if( a!=1 )
    {
        force_ops[0].second.push_back(std::make_pair( oldF, 1-a ));
    }

    // deltaV = newV - oldV
    velocity_ops[0].first = _deltaV;
    velocity_ops[0].second.push_back(std::make_pair( newV, 1.0 ));
    velocity_ops[0].second.push_back(std::make_pair( oldV, -1.0 ));


    // compute average_f (only for independent dofs)
    sop.vop.v_multiop( force_ops );


    // compute deltaV (propagated to mapped dofs)
    simulation::MechanicalVMultiOpVisitor multivis( &sop.mparams(), velocity_ops );
    multivis.mapped = true; // propagating
    this->getContext()->executeVisitor( &multivis, true );


    // compute rhs from average_f, average_v and deltaV
    sop.vop.v_teq(residual,-h); // -h*average_f
    sop.mop->setDx(_deltaV);
    sop.mop.addMBKdx(residual,1,0,0); // M*deltaV - h*average_f

    // compute and return infinite norm of the residual
    sop.vop.v_norm(residual,0);
    SReal e = sop.vop.finish();

    if( sys.n )
    {

        // compute phi at the end of the step
        vec phi(sys.n);
        rhs_constraints_dynamics(phi, sys, newX, newV );


        // compute C at the end of the step ??
        // compute C.lambda - phi
        *residual_constraints = - sys.C * lambda - phi;

        // project unilateral error
        unsigned off=0;
        for(unsigned i = 0, end = sys.compliant.size(); i < end; ++i) // loop over compliant dofs
        {
            system_type::dofs_type* dofs = sys.compliant[i];
            const unsigned dim = dofs->getSize(); // nb constraints
            const unsigned constraint_dim = dofs->getDerivDimension(); // nb lines per constraint
            const unsigned size = dim*constraint_dim;

            if( m_projectors[i] ) // non bilateral constraints
            {
                SReal* violation = new SReal[size];
                dofs->copyToBuffer(violation, newX.getId(dofs), size);

                for( unsigned j=0 ; j<dim ; ++j )
                {
                    if( violation[constraint_dim*j]>=0 ) // not violated unilateral -> no error
                    {
                        for( unsigned k=0 ; k<constraint_dim ; ++k )
                            (*residual_constraints)( off + constraint_dim*j + k ) = 0;
                    }
                }

                delete [] violation;
            }

            off += size;
        }

        e = std::max( e, residual_constraints->lpNorm<Eigen::Infinity>() );

    }


    if(debug.getValue())
    {
        std::cout<<std::endl;
        sop.vop.print(newV,std::cout,"CompliantNLImplicitSolver::compute_residual, newV= ", "\n");
        sop.vop.print(newX,std::cout,"CompliantNLImplicitSolver::compute_residual, newX= ", "\n");
        sop.vop.print(lagrange,std::cout,"CompliantNLImplicitSolver::compute_residual, lagrange= ", "\n");
        sop.vop.print(residual,std::cout,"CompliantNLImplicitSolver::compute_residual, err= ", " | ");
        if(residual_constraints) std::cout<<residual_constraints->transpose();
        std::cout<<" -> "<<e;
        std::cout<<std::endl;
    }

    return e;
}

void CompliantNLImplicitSolver::compute_jacobian(SolverOperations sop)
{
//    cerr<<"compute_jacobian, h = " << sop.mparams().dt() <<endl;
//    cerr<<"compute_jacobian, kfactor = " << sop.mparams().kFactor() <<endl;
//    cerr<<"compute_jacobian, x = "; sop.vop.print(sop.mparams().x(),cerr); cerr<<endl;
//    cerr<<"compute_jacobian, v = "; sop.vop.print(sop.mparams().v(),cerr); cerr<<endl;

    simulation::MechanicalComputeGeometricStiffness gsvis( &sop.mparams(), core::VecDerivId::force() );
    send( gsvis );

    // assemble system
    perform_assembly( &sop.mparams(), sys );

    // hack for unilateral constraints
    // handle them as an active set of bilateral constraints
    handleUnilateralConstraints();


    // debugging
//    if( debug.getValue() ) sys.debug();
    //    cerr<<"compute_jacobian, H = "<< endl << kkt_type::system_type::dmat(sys.H) << endl;

    this->filter_constraints( sop.posId );

    // system factor
    {
        scoped::timer step("system factor");
        kkt->factor( sys );
    }

}


void CompliantNLImplicitSolver::integrate( SolverOperations& sop, core::MultiVecCoordId oldPos, core::MultiVecCoordId newPos, core::MultiVecDerivId vel )
{
    core::behavior::BaseMechanicalState::VMultiOp multi;
    multi.resize(1);
    multi[0].first = newPos;
    multi[0].second.push_back( std::make_pair(oldPos, 1.0) );
    multi[0].second.push_back( std::make_pair(vel, beta.getValue() * sop.mparams().dt() ) );
    sop.vop.v_multiop( multi );
}


void CompliantNLImplicitSolver::v_eq_all(const core::ExecParams* params, sofa::core::MultiVecId v, sofa::core::ConstMultiVecId a) // v=a including mapped dofs
{
    simulation::MechanicalVOpVisitor vis( params, v, a );
    vis.mapped = true;
    this->getContext()->executeVisitor( &vis, true );
}


void CompliantNLImplicitSolver::handleUnilateralConstraints()
{
    if( sys.n )
    {
        m_projectors.resize(sys.compliant.size());
        for( size_t i=0 ; i<sys.compliant.size() ; ++i )
        {
            m_projectors[i] = sys.constraints[i].projector;
            sys.constraints[i].projector = NULL;
        }
    }
}

void CompliantNLImplicitSolver::solve(const core::ExecParams* eparams,
                         SReal dt,
                         core::MultiVecCoordId posId,
                         core::MultiVecDerivId velId)
{
    assert(kkt);

    static_cast<simulation::Node*>(getContext())->precomputeTraversalOrder( eparams );

    SolverOperations sop( eparams, this->getContext(), alpha.getValue(), beta.getValue(), dt, posId, velId, true/*, staticSolver.getValue()*/ );

    if( iterations.getValue() <= 1 )
    {
        // regular, linearized solution
        firstGuess( sop, posId, velId );
        // integrate positions posId += h.newV
        integrate( sop, posId, posId, velId );
    }
    else
    {

        // state copy at the beginning of the time step
        _x0.realloc( &sop.vop ); // position at the beginning of the time step
        _v0.realloc( &sop.vop, false, true ); // velocity at the beginning of the time step
        _f0.realloc( &sop.vop ); // force at the beginning of the time step - TODO do not allocate it if alpha=1
        _vfirstguess.realloc( &sop.vop ); // velocity computed by first guess
        // linear system variables (for a Newton iteration)
        _err.realloc( &sop.vop, false, true ); // residual
        // rhs variables
        _deltaV.realloc( &sop.vop, false, true ); // newV - v

        // evolving state during Newton iterations (directly working on current multivec)
        MultiVecCoord newX( &sop.vop, posId );
        MultiVecDeriv newV( &sop.vop, velId );
        MultiVecDeriv newF( &sop.vop, core::VecDerivId::force() );



        // copying velocity at the beginning of the time step
        // v0 = v including mapped dofs
        v_eq_all( &sop.mparams(), _v0, velId );
        // x0 = x only for independent dofs
        sop.vop.v_eq( _x0, posId );


        // allocating eventual lagrange multipliers
//        if( sys.n ) // no, sys is not built here
        {
            scoped::timer step("lambdas alloc");
            lagrange.realloc( &sop.vop, false, true );

            lagrange.clear(); // no warmstart for lagrange multipliers, the Newton iterations should compensate the geometric stiffness for compliance

            _lambdafirstguess.realloc( &sop.vop, false, true ); // lambdas computed by first guess
        }

        // first approximation using a regular, linearized system
        firstGuess( sop, newX, newV );


        if(debug.getValue())
        {
            sop.vop.print(newV.id(),std::cout,"CompliantNLImplicitSolver::solve, firstGuess newV= ","\n");
            sop.vop.print(lagrange.id(),std::cout,"CompliantNLImplicitSolver::solve, firstGuess lambdas= ","\n");
        }

        // keeping linear solution as a failsafe
        sop.vop.v_eq( _vfirstguess, newV );

        if( sys.n ) v_eq_all( &sop.mparams(), _lambdafirstguess, lagrange );
        // TODO multiop these copies

        // blending with force at the beginning of the step -> need to compute f0
        if( alpha.getValue() != 1 )
        {
            // as there is no stabilization, position are not moved so initial forces computed in firstGuess are still coherent
            sop.vop.v_eq( _f0, core::VecDerivId::force() );
        }



        // updating newX and deltaV
        // computing newF (also important for geometric stiffness)
        // compute rhs and residual norm
        vec lambda(sys.n);
        sys.copyFromCompliantMultiVec( lambda, lagrange );


        vec x(sys.size()); // unknown
        vec residual(sys.size()); // residual
        std::unique_ptr<chuck_type> residual_constraints( sys.n?new chuck_type(&residual(sys.m),sys.n):NULL);

        handleUnilateralConstraints();

        SReal resnorm = compute_residual( sop, _err, newX, newV, newF, _x0, _v0, _f0, lambda, residual_constraints.get() );
//        if( debug.getValue() ){
//            cerr<<"CompliantNLImplicitSolver::solveStep, resnorm = " << resnorm << ", err = " << err << endl;
//        }

        SReal stpmax = 0; // can be used by line-search


        SReal localprecision = precision.getValue();
        if( relative.getValue() ) localprecision *= resnorm; // to be cleaner, the norm using the state at the beggining of the time step should be used

        SReal resnormold = resnorm;



        unsigned num=1;
        for( ;  num<iterations.getValue() && resnorm > localprecision ; num++ )
        {

            SReal resnormit = resnorm;

//            sop.vop.print(newX.id(),std::cout,"CompliantNLImplicitSolver::solve, newX= ","\n");

    //        cerr<<"solve0 = " << sop.mparams().kFactor() <<endl;
            compute_jacobian(sop); // at x+h.newV, newV is the last state propagated in compute_residual





            // system solution / rhs
            x = vec::Zero(sys.size());
            sys.copyFromMultiVec( residual, _err );      // just the ode part

            if(debug.getValue()) std::cerr<<"CompliantNLImplicitSolver::solve, residual= "<<residual.transpose()<<"\n";

            kkt->solve(x, sys, residual);


            if(debug.getValue()) std::cerr<<"CompliantNLImplicitSolver::solve, sol= "<<x.transpose()<<"\n";


            // the solution given by the newton iteration needs to be projected
            // correction = substract of the solution
            x.head(sys.m) = - sys.P * x.head(sys.m);


            if( newtonStepLength.getValue() == 1 )
            {
                if( num==1 ) // first iteration
                {
                    static const SReal STPMX = 100.0;

//                    vec v0(sys.size());
//                    sys.copyFromMultiVec( v0, _v0 );
//                     Note the dot product cannot be performed on multivec, to take into account constraint dofs

//                    stpmax = STPMX * std::max( std::sqrt( v0.dot(v0) ), SReal(sys.size()) );

                    // we do not want to take into account constraint dofs that correspond to the lamda at the beginning of the step == 0
                    sop.vop.v_dot(_v0,_v0);
                    stpmax = STPMX * std::max( SReal(std::sqrt( sop.vop.finish() )), SReal(sys.m) );
                }


                // line search to find a good sub-part of the correction that decreased "sufficiently" the error
                if( lnsrch( resnorm, x, residual, stpmax, sop, _err, newX, newV, newF, _x0, _v0, _f0 ) )
                    break; // there is now way to improve convergence
            }
            else
            {
                // correction = subpart of the solution
                x = x * newtonStepLength.getValue();

                // applying correction
                sys.addToMultiVec( newV, x );
                if( sys.n )
                {
                    lambda += x.tail(sys.n);
                    sys.copyToCompliantMultiVec( lagrange, lambda );
                }

                resnorm = compute_residual(sop,_err,newX,newV,newF,_x0,_v0,_f0,lambda,residual_constraints.get());

                if( resnorm>resnormit )
                {
//                    std::cerr<<"NO SUBSTEPS : "<<resnorm<<" "<<resnormit<<std::endl;
                    // there is now way to improve convergence
                    // removing "correction"
                    sys.addToMultiVec( newV, -x );
                    if( sys.n )
                    {
                        lambda -= x.tail(sys.n);
                        sys.copyToCompliantMultiVec( lagrange, lambda );
                    }
                    break;
                }

                // applying the newton sub-steps while it is converging
                // simple way to improve convergence in really non-linear region
                // could be improved a lot with more complex strategies
                for( int i=1 ; i<1.0/newtonStepLength.getValue() ; ++i )
                {
                     SReal prevresnorm = resnorm;
                     sys.addToMultiVec( newV, x );
                     if( sys.n )
                     {
                         lambda += x.tail(sys.n);
                         sys.copyToCompliantMultiVec( lagrange, lambda );
                     }

                     resnorm = compute_residual(sop,_err,newX,newV,newF,_x0,_v0,_f0,lambda,residual_constraints.get());
                     if( resnorm>prevresnorm )
                     {
//                         std::cerr<<"TOO MUCH SUBSTEPS : "<<prevresnorm<<" "<<resnorm<<std::endl;
                         sys.addToMultiVec( newV, -x );

                         if( sys.n )
                         {
                             lambda -= x.tail(sys.n);
                             sys.copyToCompliantMultiVec( lagrange, lambda );
                         }

                         resnorm = compute_residual(sop,_err,newX,newV,newF,_x0,_v0,_f0,lambda,residual_constraints.get()); // needs to be performed again to have right forces

                         break;
                     }
                }


            }

            if( debug.getValue() ) {
//                std::cerr << "next residual= " << resnorm << endl; //":\t" << residual.transpose() << std::endl;
                cerr<<"end of iteration " << num << " ==================================== " << endl;
            }

            this->clear_constraints();

        }

        if( resnorm > resnormold )
        {
            // fail-safe solution computed by first guess
            sop.vop.v_eq( velId, _vfirstguess );
//            std::cerr<<"DID NOT CONVERGE "<<resnorm<<" "<<resnormold<<std::endl;
            if( sys.n ) v_eq_all( &sop.mparams(), lagrange, _lambdafirstguess );
        }
//        else
//        {
////            std::cerr<<"CONVERGE "<<num<<" "<<resnorm<<" "<<resnormold<<std::endl;
//            set_state_lambda( sys, x.tail(sys.n) ); // setting lambda
//        }

        // integrate positions posId = x0 + h.newV
        integrate( sop, posId, _x0, velId );

    }


    // propagate lambdas if asked to
    unsigned constraint_f = this->constraint_forces.getValue().getSelectedId();
    if( constraint_f && sys.n ) {
        scoped::timer step("constraint_forces");
        simulation::propagate_constraint_force_visitor prop( &sop.mparams(), core::VecId::force(), lagrange.id(), formulation.getValue().getSelectedId()==FORMULATION_VEL ? 1.0/sys.dt : 1.0, constraint_f>1, constraint_f==3 );
        send( prop );
    }

    // POST STABILIZATION
    if( stabilization.getValue().getSelectedId() )
    {
        if( stabilization.getValue().getSelectedId()!=POST_STABILIZATION_ASSEMBLY )
        {
            serr<<"Only full post-stabilization "<<POST_STABILIZATION_ASSEMBLY<<" can be (and will be) computed"<<sendl;
            stabilization.beginWriteOnly()->setSelectedItem(POST_STABILIZATION_ASSEMBLY); stabilization.endEdit();
        }
        post_stabilization( sop, posId, velId, true, true );
    }

}





bool CompliantNLImplicitSolver::lnsrch( SReal& resnorm, vec& p, vec& residual, SReal stpmax, SolverOperations sop, core::MultiVecDerivId err, core::MultiVecCoordId newX, const core::MultiVecDerivId newV, core::MultiVecDerivId newF, core::MultiVecCoordId oldX, core::MultiVecDerivId oldV, core::MultiVecDerivId oldF )
{
    static const SReal ALF=1.0e-4, TOLX=std::numeric_limits<SReal>::epsilon();
    static const SReal MINIMALSTEP = exp(log(std::numeric_limits<SReal>::epsilon())/4.0); // to increase the smallest amount of correction
    unsigned int i;
    SReal a,alam,alam2=0.0,alamin,b,disc,resnorm2=0.0;
    SReal rhs1,rhs2,slope,tmplam;

    SReal resnormold = resnorm;

    const unsigned n = sys.size();

    std::unique_ptr<chuck_type> residual_constraints( sys.n?new chuck_type(&residual(sys.m),sys.n):NULL);

    vec x(n), xold(n);
    sys.copyFromMultiVec( xold, newV );
    sys.copyFromCompliantMultiVec( xold, lagrange );


    // scale if attempted step is too big
    {
        SReal sum = std::sqrt( p.dot(p) );
        if( sum > stpmax ) p *= stpmax/sum;
    }

    {
        SReal test=0.0, temp;
        for( i=0 ; i<n ; i++ ) {
            temp = fabs(p[i]) / std::max( std::abs(xold[i]), SReal(1.0) );
            if( temp > test ) test=temp;
        }
        alamin = std::max( TOLX/test, MINIMALSTEP );
    }

    slope = residual.dot( p );
    if( slope >= 0.0 )
    {
        if( debug.getValue() )
        {
            std::cerr<<SOFA_CLASS_METHOD<<"Roundoff problem (slope="<<slope<<")\n";
            //        return true; // it won't converge anyway
            std::cerr<<SOFA_CLASS_METHOD<<"Trying failsafe sub-stepping (step="<<alamin<<")\n";
        }

        // hack not to perform to much sub-steps
        alamin = std::min( SReal(0.1), SReal(alamin*100) );


        // applying correction
        x = xold+alamin*p;
        sys.copyToMultiVec( newV, x );
        sys.copyToCompliantMultiVec( lagrange, x );

        resnorm = compute_residual(sop,err,newX,newV,newF,oldX,oldV,oldF,x.tail(sys.n),residual_constraints.get());

        if( resnorm>resnormold )
        {
//            std::cerr<<"NO SUBSTEPS : "<<resnorm<<" "<<resnormold<<std::endl;
            // there is now way to improve convergence
            // removing "correction"
            sys.copyToMultiVec( newV, xold );
            sys.copyToCompliantMultiVec( lagrange, xold );
            return true;
        }

        // applying the newton sub-steps while it is converging
        // simple way to improve convergence in really non-linear region
        // could be improved a lot with more complex strategies
        for( int i=1 ; i<1.0/alamin ; ++i )
        {
             SReal prevresnorm = resnorm;
             x = xold+alamin*(i+1)*p;
             sys.copyToMultiVec( newV, x );
             sys.copyToCompliantMultiVec( lagrange, x );

             resnorm = compute_residual(sop,err,newX,newV,newF,oldX,oldV,oldF,x.tail(sys.n),residual_constraints.get());
             if( resnorm>prevresnorm )
             {
//                 std::cerr<<"TOO MUCH SUBSTEPS : "<<prevresnorm<<" "<<resnorm<<std::endl;

                 x = xold+alamin*i*p;
                 sys.copyToMultiVec( newV, x );
                 sys.copyToCompliantMultiVec( lagrange, x );

                 resnorm = compute_residual(sop,err,newX,newV,newF,oldX,oldV,oldF,x.tail(sys.n),residual_constraints.get()); // needs to be performed again to have right forces

//                 std::cerr<<"Newton sub-steps: "<<i*newtonStepLength.getValue()<<std::endl;

                 break;
             }
        }



        return false;

    }




    alam=1.0;
    for (;;) {
        x = xold+alam*p;
        sys.copyToMultiVec( newV, x );
        sys.copyToCompliantMultiVec( lagrange, x );

        resnorm = compute_residual(sop,err,newX,newV,newF,oldX,oldV,oldF,x.tail(sys.n),residual_constraints.get());
//        std::cerr<<SOFA_CLASS_METHOD<<"alam = "<<alam<<" "<<resnorm<<" "<<resnormold<<" "<<resnormold+ALF*alam*slope<<std::endl;

        if (resnorm <= resnormold+ALF*alam*slope)
                {
//                    /*if( alam!=1 ) */std::cerr<<SOFA_CLASS_METHOD<<"resnorm <= resnormold "<<resnorm<<" "<<resnormold<<" "<<resnormold+ALF*alam*slope<<std::endl;
                    return false;
                }
        else if (alam < alamin)
        {
//            sys.copyToMultiVec( newV, xold );
//            sys.copyToCompliantMultiVec( lagrange, xold );
            // convergence is achieved, no need for updating internal force, no jacobian will be evaluated
            resnorm = resnormold;
//            std::cerr<<SOFA_CLASS_METHOD<<"alam < alamin "<<alam<<" "<<alamin<<std::endl;
            return true;

            // hack that seems more stable
            // always add a small portion of correction and contine newton iterations (stopping can bring in crappy configurations that still improve the initial error but are far from valid)
//            x = xold+alamin*p;
//            sys.copyToMultiVec( newV, x );
//            resnorm = compute_residual(sop,err,newX,newV,newF,oldX,oldV,oldF);
//            return false;

        }
        else {
            if (alam == 1.0)
                tmplam = -slope/(2.0*(resnorm-resnormold-slope));
            else {
                rhs1=resnorm-resnormold-alam*slope;
                rhs2=resnorm2-resnormold-alam2*slope;
                a=(rhs1/(alam*alam)-rhs2/(alam2*alam2))/(alam-alam2);
                b=(-alam2*rhs1/(alam*alam)+alam*rhs2/(alam2*alam2))/(alam-alam2);
                if (a == 0.0) tmplam = -slope/(2.0*b);
                else {
                    disc=b*b-3.0*a*slope;
                    if (disc < 0.0) tmplam=0.5*alam;
                    else if (b <= 0.0) tmplam=(-b+sqrt(disc))/(3.0*a);
                    else tmplam=-slope/(b+sqrt(disc));
                }
                if (tmplam>0.5*alam)
                    tmplam=0.5*alam;
            }
        }
        alam2=alam;
        resnorm2 = resnorm;
        alam=std::max(tmplam,SReal(0.1)*alam);
    }
}


void CompliantNLImplicitSolver::compute_forces(SolverOperations& sop, core::behavior::MultiVecDeriv& f, core::behavior::MultiVecDeriv* f_k )
{
//    if( !staticSolver.getValue() )
        return CompliantImplicitSolver::compute_forces( sop, f, f_k );
//    else
//    {
//        std::cerr<<SOFA_CLASS_METHOD<<"Static solver does not work!\n";

////        scoped::timer step("implicit rhs computation");

////        {
////            scoped::timer substep("forces computation");
////            sop.mop.computeForce( f ); // f = fk
////        }

////        {
////            scoped::timer substep("c_k = -fk");
////            c.eq( f, -1 );
////        }

////        if( !neglecting_compliance_forces_in_geometric_stiffness.getValue() )
////        {
////            scoped::timer substep("f += fc");

////            // using lambdas computed at the previous time step as an approximation of the forces generated by compliances
////            // If no approximation are known, 0 is used and the associated compliance won't contribute
////            // to geometric sitffness generation for this step.
////            simulation::MechanicalAddComplianceForce lvis( &sop.mparams(), f, lagrange, 1 ); // f += fc   f += lambda / dt
////            send( lvis );

////            // TODO have a look about reseting or not forces of mapped dofs
////        }
////
////        // computing mapping geometric stiffnesses based on child force stored in f
////        simulation::MechanicalComputeGeometricStiffness( &sop.mparams(), f );
//    }
}

void CompliantNLImplicitSolver::firstGuess( SolverOperations& sop, core::MultiVecCoordId posId, core::MultiVecDerivId velId )
{
    MultiVecDeriv f( &sop.vop, core::VecDerivId::force() ); // total force (stiffness + compliance)
    MultiVecDeriv v( &sop.vop, velId );
    _ck.realloc( &sop.vop, false, true ); // the right part of the implicit system (c_k term)

    unsigned int form = formulation.getValue().getSelectedId();
    bool useVelocity = (form==FORMULATION_VEL);

    if( !useVelocity ) _acc.realloc( &sop.vop );

    // compute forces and implicit right part (c_k term, homogeneous to a momentum)
    // warning: must be call before assemblyVisitor since the mapping's geometric
    // stiffness depends on its child force
    compute_forces( sop, f, &_ck );


    // perform system assembly & factorization
//    compute_jacobian(sop);
    perform_assembly( &sop.mparams(), sys );
    this->filter_constraints( sop.posId );
    kkt->factor( sys );


    if( sys.n )
    {
        scoped::timer step("lambdas alloc");
        lagrange.realloc( &sop.vop, false, true );
    }

    // system solution / rhs
    vec x(sys.size());
    vec rhs(sys.size());

    // ready to solve
    {
        scoped::timer step("dynamics system solve");

        if( warm_start.getValue() ) get_state( x, sys, useVelocity ? velId : _acc.id() );
        else x = vec::Zero( sys.size() );

        rhs_dynamics( sop, rhs, sys, _ck, posId, velId );

        kkt->solve(x, sys, rhs);

        switch( form )
        {
        case FORMULATION_VEL: // p+ = p- + h.v
            set_state( sys, x, velId ); // set v and lambda
            break;
        case FORMULATION_DV: // v+ = v- + dv     p+ = p- + h.v
            set_state( sys, sys.P * x, _acc.id() ); // set v and lambda
            v.peq( _acc.id() );
            break;
        case FORMULATION_ACC: // v+ = v- + h.a   p+ = p- + h.v
            set_state( sys, sys.P * x, _acc.id() ); // set v and lambda
            v.peq( _acc.id(), sop.mparams().dt() );
            break;
        }

//            std::cerr<<"CompliantNLImplicitSolver::firstGuess lambdas : "<< x.tail(sys.n).transpose() <<std::endl;

        this->clear_constraints();

    }

 }



}
}
}
