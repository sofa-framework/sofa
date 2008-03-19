#include <CudaMasterContactSolver.h>

#include <sofa/helper/LCPcalc.h>
#include <sofa/core/ObjectFactory.h>

using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::defaulttype;
using namespace helper::system::thread;
using namespace core::componentmodel::behavior;

#define MAX_NUM_CONSTRAINTS 3000
//#define DISPLAY_TIME

CudaMasterContactSolver::CudaMasterContactSolver()
    :initial_guess_d(initData(&initial_guess_d, true, "initial_guess","activate LCP results history to improve its resolution performances."))
    //,tol_d( initData(&tol_d, 0.001, "tolerance", ""))
    //,maxIt_d( initData(&maxIt_d, 1000, "maxIt", ""))
    //,mu_d( initData(&mu_d, 0.6, "mu", ""))
    ,useGPU_d(initData(&useGPU_d, 6, "useGPU", "compute LCP using GPU"))
    ,_mu(0.0)
{

    _W.resize(MAX_NUM_CONSTRAINTS,MAX_NUM_CONSTRAINTS);
    _A.resize(MAX_NUM_CONSTRAINTS,2*MAX_NUM_CONSTRAINTS+1);
    _dFree.resize(MAX_NUM_CONSTRAINTS);
    _f.resize(MAX_NUM_CONSTRAINTS+1);
    _res.resize(MAX_NUM_CONSTRAINTS+1);
    _numConstraints = 0;
    _mu = 0.0;

    _numPreviousContact=0;
    _PreviousContactList = (contactBuf *)malloc(MAX_NUM_CONSTRAINTS * sizeof(contactBuf));
    _cont_id_list = (long *)malloc(MAX_NUM_CONSTRAINTS * sizeof(long));
}

void CudaMasterContactSolver::init()
{
    getContext()->get<core::componentmodel::behavior::BaseConstraintCorrection>(&constraintCorrections, core::objectmodel::BaseContext::SearchDown);
}

void CudaMasterContactSolver::build_LCP()
{
    _numConstraints = 0;

    simulation::tree::MechanicalResetConstraintVisitor().execute(context);
    simulation::tree::MechanicalAccumulateConstraint(_numConstraints, _mu).execute(context);

    if (_numConstraints > MAX_NUM_CONSTRAINTS)
    {
        cerr<<endl<<"Error in MasterContactSolver, maximum number of contacts exceeded, "<< _numConstraints/3 <<" contacts detected"<<endl;
        exit(-1);
    }
    _dFree.resize(_numConstraints);
    _W.setwarpsize(_mu);
    _W.resize(_numConstraints,_numConstraints);

    CudaMechanicalGetConstraintValueVisitor(&_dFree).execute(context);

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->getCompliance(&_W);
    }

    if (initial_guess_d.getValue())
    {
        CudaMechanicalGetContactIDVisitor(_cont_id_list).execute(context);
        computeInitialGuess();
    }
}

void CudaMasterContactSolver::computeInitialGuess()
{
    int numContact = (_mu > 0.0) ? _numConstraints/3 : _numConstraints;

    for (int c=0; c<numContact; c++)
    {
        if (_mu>0.0)
        {
            _f[3*c  ] = 0.0;
            _f[3*c+1] = 0.0;
            _f[3*c+2] = 0.0;
        }
        else
        {
            _f[c]=  0.0;
        }
    }

    for (int c=0; c<numContact; c++)
    {
        for (unsigned int pc=0; pc<_numPreviousContact; pc++)
        {
            if (_cont_id_list[c] == _PreviousContactList[pc].id)
            {
                if (_mu>0.0)
                {
                    _f[3*c  ] = _PreviousContactList[pc].F.x();
                    _f[3*c+1] = _PreviousContactList[pc].F.y();
                    _f[3*c+2] = _PreviousContactList[pc].F.z();
                }
                else
                {
                    _f[c] =  _PreviousContactList[pc].F.x();
                }
            }
        }
    }
}

void CudaMasterContactSolver::keepContactForcesValue()
{
    _numPreviousContact=0;

    int numContact = (_mu > 0.0) ? _numConstraints/3 : _numConstraints;

    for (int c=0; c<numContact; c++)
    {
        if (_mu>0.0)
        {
            if (_f[3*c]>0.0)
            {
                _PreviousContactList[_numPreviousContact].id = (_cont_id_list[c] >= 0) ? _cont_id_list[c] : -_cont_id_list[c];
                _PreviousContactList[_numPreviousContact].F.x() = _f[3*c];
                _PreviousContactList[_numPreviousContact].F.y() = _f[3*c+1];
                _PreviousContactList[_numPreviousContact].F.z() = _f[3*c+2];
                _numPreviousContact++;
            }
        }
        else
        {
            if (_f[c]>0.0)
            {
                _PreviousContactList[_numPreviousContact].id = (_cont_id_list[c] >= 0) ? _cont_id_list[c] : -_cont_id_list[c];
                _PreviousContactList[_numPreviousContact].F.x() = _f[c];
                _numPreviousContact++;
            }
        }
    }
}

void CudaMasterContactSolver::step(double dt)
{
    //float _tol = (float) tol_d.getValue();
    //int _maxIt = maxIt_d.getValue();
    //_mu = mu.getValue();
    float _tol = 0.001;
    int _maxIt = 1000;

    context = dynamic_cast<simulation::tree::GNode *>(this->getContext()); // access to current node

#ifdef DISPLAY_TIME
    CTime *timer = new CTime();
    double time = 0;
    time = (double) timer->getTime();
#endif

    for(simulation::tree::GNode::ChildIterator it=context->child.begin(); it!=context->child.end(); ++it)
    {
        for ( unsigned i=0; i<(*it)->behaviorModel.size(); i++)
        {
            (*it)->behaviorModel[i]->updatePosition(dt);
        }
    }

    simulation::tree::MechanicalBeginIntegrationVisitor beginVisitor(dt);
    context->execute(&beginVisitor);

    for(simulation::tree::GNode::ChildIterator it=context->child.begin(); it!=context->child.end(); ++it)
    {
        for ( unsigned i=0; i<(*it)->solver.size(); i++)
        {
            (*it)->solver[i]->solve(dt);
        }
    }
    simulation::tree::MechanicalPropagateFreePositionVisitor().execute(context);

    core::componentmodel::behavior::BaseMechanicalState::VecId dx_id = core::componentmodel::behavior::BaseMechanicalState::VecId::dx();
    simulation::tree::MechanicalVOpVisitor(dx_id).execute( context);
    simulation::tree::MechanicalPropagateDxVisitor(dx_id).execute( context);
    simulation::tree::MechanicalVOpVisitor(dx_id).execute( context);

#ifdef DISPLAY_TIME
    std::cout<<" Free Motion " << ( (double) timer->getTime() - time)*0.001 <<" ms" <<std::endl;
    time = (double) timer->getTime();
#endif

    computeCollision();

#ifdef DISPLAY_TIME
    std::cout<<" computeCollision " << ( (double) timer->getTime() - time)*0.001 <<" ms" <<std::endl;
    time = (double) timer->getTime();
#endif

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }

    build_LCP();

#ifdef DISPLAY_TIME
    std::cout<<" build_LCP " << ( (double) timer->getTime() - time)*0.001<<" ms" <<std::endl;
    time = (double) timer->getTime();
#endif

    if (! initial_guess_d.getValue()) _f.clear();

    if (_mu > 0.0)
    {
        sofa::gpu::cuda::CudaLCP::CudaNlcp_gaussseidel(useGPU_d.getValue(),_numConstraints, _dFree.getCudaVector(), _W.getCudaMatrix(), _f.getCudaVector(),_res.getCudaVector(), _mu,_tol, _maxIt);
    }
    else
    {
        sofa::gpu::cuda::CudaLCP::CudaGaussSeidelLCP1(useGPU_d.getValue(),_numConstraints, _dFree.getCudaVector(), _W.getCudaMatrix(), _f.getCudaVector(),_res.getCudaVector(), _tol, _maxIt);
    }

#ifdef DISPLAY_TIME
    std::cout<<" solve_LCP" <<( (double) timer->getTime() - time)*0.001<<" ms" <<std::endl;
    time = (double) timer->getTime();
#endif

    if (initial_guess_d.getValue()) keepContactForcesValue();

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->applyContactForce(&_f);
    }

    simulation::tree::MechanicalPropagateAndAddDxVisitor().execute( context);

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }

#ifdef DISPLAY_TIME
    std::cout<<" contactCorrections" <<( (double) timer->getTime() - time)*0.001 <<" ms" <<std::endl;
#endif

    simulation::tree::MechanicalEndIntegrationVisitor endVisitor(dt);
    context->execute(&endVisitor);
}

SOFA_DECL_CLASS(CudaMasterContactSolver)

int CudaMasterContactSolverClass = core::RegisterObject("Cuda Constraint solver")
        .add< CudaMasterContactSolver >()
        ;

} // namespace odesolver

} // namespace component

} // namespace sofa
