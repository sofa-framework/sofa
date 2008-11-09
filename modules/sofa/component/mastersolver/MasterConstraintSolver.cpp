#include <sofa/component/mastersolver/MasterConstraintSolver.h>
#include <sofa/component/mastersolver/MasterContactSolver.h>

#include <sofa/simulation/common/AnimateVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/SolveVisitor.h>

#include <sofa/helper/LCPcalc.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/helper/system/thread/CTime.h>

#include <math.h>
#include <iostream>
#include <map>



namespace sofa
{

namespace component
{

namespace mastersolver
{

using namespace sofa::component::odesolver;
using namespace sofa::defaulttype;
using namespace helper::system::thread;
using namespace core::componentmodel::behavior;



MasterConstraintSolver::MasterConstraintSolver()
    :displayTime(initData(&displayTime, false, "displayTime","Display time for each important step of MasterConstraintSolver.")),
     _tol( initData(&_tol, 0.00001, "tolerance", "Tolerance of the Gauss-Seidel")),
     _maxIt( initData(&_maxIt, 1000, "maxIterations", "Maximum number of iterations of the Gauss-Seidel"))
{
}

MasterConstraintSolver::~MasterConstraintSolver()
{
}

void MasterConstraintSolver::init()
{
    getContext()->get<core::componentmodel::behavior::BaseConstraintCorrection> ( &constraintCorrections, core::objectmodel::BaseContext::SearchDown );
}

void MasterConstraintSolver::step ( double dt )
{

    CTime *timer;
    double time = 0.0;
    double timeScale = 1.0 / (double)CTime::getRefTicksPerSec();
    if ( displayTime.getValue() )
    {
        timer = new CTime();
        time = (double) timer->getTime();
        std::cout<<"********* Start Iteration in MasterConstraintSolver::step *********" <<std::endl;
    }

    bool debug =false;
    if (debug)
        std::cerr<<"MasterConstraintSolver::step is called"<<std::endl;
    simulation::tree::GNode *context = dynamic_cast<simulation::tree::GNode *>(this->getContext()); // access to current node

    // Update the BehaviorModels
    // Required to allow the RayPickInteractor interaction
    for (simulation::tree::GNode::ChildIterator it = context->child.begin(); it != context->child.end(); ++it)
    {
        for (unsigned i=0; i<(*it)->behaviorModel.size(); i++)
            (*it)->behaviorModel[i]->updatePosition(dt);
    }
    if (debug)
        std::cerr<<"Free Motion is called"<<std::endl;

    ///////////////////////////////////////////// FREE MOTION /////////////////////////////////////////////////////////////
    simulation::MechanicalBeginIntegrationVisitor beginVisitor(dt);
    context->execute(&beginVisitor);
    for (simulation::tree::GNode::ChildIterator it = context->child.begin(); it != context->child.end(); ++it)
    {
        for (unsigned i=0; i<(*it)->solver.size(); i++)
            (*it)->solver[i]->solve(dt, core::componentmodel::behavior::BaseMechanicalState::VecId::freePosition(), core::componentmodel::behavior::BaseMechanicalState::VecId::freeVelocity());
    }

    simulation::MechanicalPropagateFreePositionVisitor().execute(context);

    core::componentmodel::behavior::BaseMechanicalState::VecId dx_id = core::componentmodel::behavior::BaseMechanicalState::VecId::dx();
    simulation::MechanicalVOpVisitor(dx_id).execute(context);
    simulation::MechanicalPropagateDxVisitor(dx_id).execute(context);
    simulation::MechanicalVOpVisitor(dx_id).execute(context);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if ( displayTime.getValue() )
    {
        std::cout << " >>>>> Begin display MasterContactSolver time" << std::endl;
        std::cout<<" Free Motion " << ( (double) timer->getTime() - time)*timeScale <<" s" <<std::endl;
        time = (double) timer->getTime();
    }

    if (debug)
        std::cerr<<"computeCollision is called"<<std::endl;

    ////////////////// COLLISION DETECTION///////////////////////////////////////////////////////////////////////////////////////////
    computeCollision();
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if ( displayTime.getValue() )
    {
        std::cout<<" computeCollision " << ( (double) timer->getTime() - time)*timeScale <<" s" <<std::endl;
        time = (double) timer->getTime();
    }

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }


    //////////////////////////////////////CONSTRAINTS RESOLUTION//////////////////////////////////////////////////////////////////////
    if (debug)
        std::cerr<<"constraints Matrix construction is called"<<std::endl;

    unsigned int numConstraints = 0;

    // mechanical action executed from root node to propagate the constraints
    simulation::MechanicalResetConstraintVisitor().execute(context);
    // calling applyConstraint
//	simulation::MechanicalAccumulateConstraint(numConstraints).execute(context);
    MechanicalSetConstraint(numConstraints).execute(context);

    // calling accumulateConstraint
    MechanicalAccumulateConstraint2().execute(context);

    if (debug)
        std::cerr<<"   1. resize constraints : numConstraints="<< numConstraints<<std::endl;

    _dFree.resize(numConstraints);
    _d.resize(numConstraints);
    _W.resize(numConstraints,numConstraints);
    _constraintsType.resize(numConstraints);
    _force.resize(numConstraints);
    _constraintsResolutions.resize(numConstraints); // _constraintsResolutions.clear();

    if (debug)
        std::cerr<<"   2. compute violation"<<std::endl;
    // calling getConstraintValue
    MechanicalGetConstraintValueVisitor(&_dFree).execute(context);

    if (debug)
        std::cerr<<"   3. get resolution method for each constraint"<<std::endl;
    // calling getConstraintResolution
    MechanicalGetConstraintResolutionVisitor(_constraintsResolutions).execute(context);

    if (debug)
        std::cerr<<"   4. get Compliance "<<std::endl;

    for (unsigned int i=0; i<constraintCorrections.size(); i++ )
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->getCompliance(&_W); // getDelassusOperator(_W) = H*C*Ht
    }

    if ( displayTime.getValue() )
    {
        std::cout<<" build problem in the constraint space " << ( (double) timer->getTime() - time)*timeScale<<" s" <<std::endl;
        time = (double) timer->getTime();
    }

    if (debug)
        std::cerr<<"Gauss-Seidel solver is called"<<std::endl;
    gaussSeidelConstraint(numConstraints, _dFree.ptr(), _W.lptr(), _force.ptr(), _d.ptr(), _constraintsResolutions);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if ( displayTime.getValue() )
    {
        std::cout<<" Solve with GaussSeidel " <<( (double) timer->getTime() - time)*timeScale<<" s" <<std::endl;
        time = (double) timer->getTime();
    }

//	helper::afficheLCP(_dFree.ptr(), _W.lptr(), _force.ptr(),  numConstraints);
//	helper::afficheLCP(_dFree.ptr(), _W.lptr(), _result.ptr(),  numConstraints);

//	fprintf(stderr, "applyContactForce\n");

    if (debug)
        std::cout<<"constraintCorrections motion is called"<<std::endl;

    ///////////////////////////////////////CORRECTIVE MOTION //////////////////////////////////////////////////////////////////////////
    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->applyContactForce(&_force);
    }

    simulation::MechanicalPropagateAndAddDxVisitor().execute(context);
    simulation::MechanicalPropagatePositionAndVelocityVisitor().execute(context);

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }

    if ( displayTime.getValue() )
    {
        std::cout<<" contactCorrections" <<( (double) timer->getTime() - time)*timeScale <<" s" <<std::endl;
        std::cout << "<<<<<< End display MasterContactSolver time." << std::endl;
    }

    simulation::MechanicalEndIntegrationVisitor endVisitor(dt);
    context->execute(&endVisitor);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}

void MasterConstraintSolver::gaussSeidelConstraint(int dim, double* dfree, double** w, double* force,
        double* d, std::vector<ConstraintResolution*>& res)
{
//	std::cout<<"------------------------------------ new iteration ---------------------------------"<<std::endl;
    int i, j, k, l, nb;

    double errF[6];
    double error=0.0;

    double tolerance = _tol.getValue();
    int numItMax = _maxIt.getValue();
    bool convergence = false;

    for(i=0; i<dim; )
    {
        res[i]->init(i, w, force);
        i += res[i]->nbLines;
    }

    for(i=0; i<numItMax; i++)
    {
        error=0.0;
        for(j=0; j<dim;)
        {
            nb = res[j]->nbLines;

            for(l=0; l<nb; l++)
            {
                errF[l] = force[j+l];
                force[j+l] = 0.0;
                d[j+l] = dfree[j+l];
            }

            // TODO : add a vector with the non null force
            for(k=0; k<dim; k++)
                for(l=0; l<nb; l++)
                    d[j+l] += w[j+l][k] * force[k];

            for(l=0; l<nb; l++)
                force[j+l] = errF[l];

            res[j]->resolution(j, w, d, force);

            if(nb > 1)
            {
                double terr = 0.0, terr2;
                for(l=0; l<nb; l++)
                {
                    terr2 = w[j+l][j+l] * (force[j+l] - errF[l]);
                    terr += terr2 * terr2;
                }
                error += sqrt(terr);
            }
            else
                error += fabs(w[j][j] * (force[j] - errF[0]));

            j += nb;
        }

        if(error < tolerance && i>0) // do not stop at the first iteration (that is used for initial guess computation)
        {
            std::cout<<" ------------------ convergence after "<<i<<" iterations ------------------"<<std::endl;
            convergence = true;
            break;
        }
    }

#ifdef DEBUG_CONVERGENCE
    static int nbFrames = 0, nbIter = 0;
    nbFrames++;
    nbIter += i+1;
    if(nbFrames>99)
    {

        std::cout << (float)nbIter/nbFrames << std::endl;
        nbFrames = 0;
        nbIter = 0;
    }
#endif

    for(i=0; i<dim; )
    {
        res[i]->store(i, force, convergence);
        int t = res[i]->nbLines;
        delete res[i];
        res[i] = NULL;
        i += t;
    }

    if(!convergence)
        std::cerr << "------------------  No convergence in gaussSeidelConstraint : error = " << error <<" ------------------" <<std::endl;
}


SOFA_DECL_CLASS ( MasterConstraintSolver )

int MasterConstraintSolverClass = core::RegisterObject ( "Constraint solver" )
        .add< MasterConstraintSolver >()
        ;

} // namespace odesolver

} // namespace component

} // namespace sofa
