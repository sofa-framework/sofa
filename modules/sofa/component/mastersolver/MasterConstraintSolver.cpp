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
    :_tol( initData(&_tol, 0.00001, "tolerance", "Tolerance of the Gauss-Seidel")),
     _mu( initData(&_mu, 0.6, "mu", "Friction coefficient")),
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
    simulation::tree::GNode *context = dynamic_cast<simulation::tree::GNode *>(this->getContext()); // access to current node

    // Update the BehaviorModels
    // Required to allow the RayPickInteractor interaction
    for (simulation::tree::GNode::ChildIterator it = context->child.begin(); it != context->child.end(); ++it)
    {
        for (unsigned i=0; i<(*it)->behaviorModel.size(); i++)
            (*it)->behaviorModel[i]->updatePosition(dt);
    }

    simulation::MechanicalBeginIntegrationVisitor beginVisitor(dt);
    context->execute(&beginVisitor);

    // Free Motion
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

    computeCollision();

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }

    unsigned int numConstraints = 0;

    // mechanical action executed from root node to propagate the constraints
    simulation::MechanicalResetConstraintVisitor().execute(context);
    double unused=0;
    // calling applyConstraint
    simulation::MechanicalAccumulateConstraint(numConstraints, unused).execute(context);

    _dFree.resize(numConstraints);
    _d.resize(numConstraints);
    _W.resize(numConstraints,numConstraints);
    _constraintsType.resize(numConstraints);
    _force.resize(numConstraints);
    _constraintsResolutions.resize(numConstraints); // _constraintsResolutions.clear();

    // calling getConstraintValue
    MechanicalGetConstraintValueVisitor(_dFree.ptr()).execute(context);
    // calling getConstraintResolution
    MechanicalGetConstraintResolutionVisitor(_constraintsResolutions).execute(context);

//	fprintf(stderr, "getCompliance\n");
    for (unsigned int i=0; i<constraintCorrections.size(); i++ )
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->getCompliance(&_W); // getDelassusOperator(_W) = H*C*Ht
    }

    gaussSeidelConstraint(numConstraints, _dFree.ptr(), _W.lptr(), _force.ptr(), _d.ptr(), _constraintsResolutions);

//	helper::afficheLCP(_dFree.ptr(), _W.lptr(), _result.ptr(),  numConstraints);

//	fprintf(stderr, "applyContactForce\n");
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

    simulation::MechanicalEndIntegrationVisitor endVisitor(dt);
    context->execute(&endVisitor);
}

void MasterConstraintSolver::gaussSeidelConstraint(int dim, double* dfree, double** w, double* force,
        double* d, std::vector<ConstraintResolution*>& res)
{
//	fprintf(stderr, "gaussSeidelConstraint\n");
    std::cout<<"------------------------------------ new iteration ---------------------------------"<<std::endl;
    int i, j, k, l, nb;

    double errF[6];
    double error=0.0;

    double tolerance = _tol.getValue();
    int numItMax = _maxIt.getValue();

    for(i=0; i<dim; )
    {
        res[i]->init(i, w);
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
            break;
        }
    }

    for(i=0; i<dim; )
    {
        int t = res[i]->nbLines;
        delete res[i];
        res[i] = NULL;
        i += t;
    }

    if(error >= tolerance)
        std::cerr << "No convergence in gaussSeidelConstraint : error = " << error << std::endl;
}


SOFA_DECL_CLASS ( MasterConstraintSolver )

int MasterConstraintSolverClass = core::RegisterObject ( "Constraint solver" )
        .add< MasterConstraintSolver >()
        ;

} // namespace odesolver

} // namespace component

} // namespace sofa
