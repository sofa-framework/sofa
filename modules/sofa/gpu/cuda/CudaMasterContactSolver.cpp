/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <CudaMasterContactSolver.h>

#include <sofa/simulation/common/AnimateVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/SolveVisitor.h>

#include <sofa/helper/LCPcalc.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/helper/system/thread/CTime.h>
#include <math.h>
#include <iostream>




namespace sofa
{

namespace component
{

namespace odesolver
{

/*
LCP::LCP(unsigned int mxC) : maxConst(mxC), tol(0.00001), numItMax(1000), useInitialF(true), mu(0.0), dim(0), lok(false)
{
	W.resize(maxConst,maxConst);
    dFree.resize(maxConst);
    f.resize(2*maxConst+1);
}

LCP::~LCP()
{
}

void LCP::reset(void)
{
	W.clear();
	W.clear();
	dFree.clear();
}
*/


using namespace sofa::defaulttype;
using namespace helper::system::thread;
using namespace core::componentmodel::behavior;


static unsigned MAX_NUM_CONSTRAINTS=2048;
//#define DISPLAY_TIME

template<class real>
CudaMasterContactSolver<real>::CudaMasterContactSolver()
    :
    useGPU_d(initData(&useGPU_d,8, "useGPU", "compute LCP using GPU"))
#ifdef CHECK
    ,check_gpu(initData(&check_gpu, true, "checkGPU", "verification of lcp error"))
#endif
    ,initial_guess(initData(&initial_guess, true, "initial_guess","activate LCP results history to improve its resolution performances."))
    ,tol( initData(&tol, 0.001, "tolerance", ""))
    ,maxIt( initData(&maxIt, 1000, "maxIt", ""))
    ,mu( initData(&mu, 0.6, "mu", ""))
    , constraintGroups( initData(&constraintGroups, "group", "list of ID of groups of constraints to be handled by this solver.") )
    ,_mu(0.6)
//, lcp1(MAX_NUM_CONSTRAINTS)
//, lcp2(MAX_NUM_CONSTRAINTS)
//, _A(&lcp1.A)
//, _W(&lcp1.W)
//, _dFree(&lcp1.dFree)
//, _result(&lcp1.f)
//, lcp(&lcp1)
{
    _W.resize(MAX_NUM_CONSTRAINTS,MAX_NUM_CONSTRAINTS);
    _dFree.resize(MAX_NUM_CONSTRAINTS);
    _f.resize(MAX_NUM_CONSTRAINTS);
    _numConstraints = 0;
    _mu = mu.getValue();
    constraintGroups.beginEdit()->insert(0);
    constraintGroups.endEdit();

    _numPreviousContact=0;
    _PreviousContactList = (contactBuf *)malloc(MAX_NUM_CONSTRAINTS * sizeof(contactBuf));
    _cont_id_list = (long *)malloc(MAX_NUM_CONSTRAINTS * sizeof(long));
}

template<class real>
void CudaMasterContactSolver<real>::init()
{
    //getContext()->get<core::componentmodel::behavior::BaseConstraintCorrection>(&constraintCorrections, core::objectmodel::BaseContext::SearchDown);

    sofa::core::objectmodel::BaseContext* context = this->getContext();
    context->get<core::componentmodel::behavior::BaseConstraintCorrection>(&constraintCorrections, core::objectmodel::BaseContext::SearchDown);
}

template<class real>
void CudaMasterContactSolver<real>::build_LCP()
{
    _numConstraints = 0;
    //sout<<" accumulateConstraint "  <<sendl;

    // mechanical action executed from root node to propagate the constraints
    simulation::MechanicalResetConstraintVisitor().execute(context);
    _mu = mu.getValue();
    simulation::MechanicalAccumulateConstraint(_numConstraints).execute(context);
    _mu = mu.getValue();


    //sout<<" accumulateConstraint_done "  <<sendl;

    if (_numConstraints > MAX_NUM_CONSTRAINTS)
    {
        serr<<endl<<"Error in CudaMasterContactSolver, maximum number of contacts exceeded, "<< _numConstraints/3 <<" contacts detected"<<endl;
        MAX_NUM_CONSTRAINTS=MAX_NUM_CONSTRAINTS+MAX_NUM_CONSTRAINTS;

        free(_PreviousContactList);
        free(_cont_id_list);

        _PreviousContactList = (contactBuf *)malloc(MAX_NUM_CONSTRAINTS * sizeof(contactBuf));
        _cont_id_list = (long *)malloc(MAX_NUM_CONSTRAINTS * sizeof(long));
    }


    if (_mu>0.0)
    {
        _dFree.resize(_numConstraints,MBSIZE);
        _f.resize(_numConstraints,MBSIZE);
        _W.resize(_numConstraints,_numConstraints,MBSIZE);
    }
    else
    {
        _dFree.resize(_numConstraints);
        _f.resize(_numConstraints);
        _W.resize(_numConstraints,_numConstraints);
    }

    _W.clear();

    CudaMechanicalGetConstraintValueVisitor(&_dFree).execute(context);
//	simulation::MechanicalComputeComplianceVisitor(_W).execute(context);

//sout<<" computeCompliance in "  << constraintCorrections.size()<< " constraintCorrections" <<sendl;

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->getCompliance(&_W);
    }
    //sout<<" computeCompliance_done "  <<sendl;

    if (initial_guess.getValue())
    {
        MechanicalGetContactIDVisitor(_cont_id_list).execute(context);
        computeInitialGuess();
    }
}

template<class real>
void CudaMasterContactSolver<real>::computeInitialGuess()
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
            _f[c] =  0.0;
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
                    _f[3*c  ] = (real)_PreviousContactList[pc].F.x();
                    _f[3*c+1] = (real)_PreviousContactList[pc].F.y();
                    _f[3*c+2] = (real)_PreviousContactList[pc].F.z();
                }
                else
                {
                    _f[c] =  (real)_PreviousContactList[pc].F.x();
                }
            }
        }
    }
}

template<class real>
void CudaMasterContactSolver<real>::keepContactForcesValue()
{
    _numPreviousContact=0;

    int numContact = (_mu > 0.0) ? _numConstraints/3 : _numConstraints;

    for (int c=0; c<numContact; c++)
    {
        if (_mu>0.0)
        {
            if (_f[3*c]>0.0)//((_result[3*c]>0.0)||(_result[3*c+1]>0.0)||(_result[3*c+2]>0.0))
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

template<class real>
void CudaMasterContactSolver<real>::step(double dt)
{

    context = dynamic_cast<simulation::Node *>(this->getContext()); // access to current node
#ifdef DISPLAY_TIME
    CTime *timer;
    double time = 0.0;
    double timeScale = 1.0 / (double)CTime::getRefTicksPerSec();
    timer = new CTime();
    time = (double) timer->getTime();
    sout<<"********* Start Iteration : " << _numConstraints << " contacts *********" <<sendl;
#endif

    // Update the BehaviorModels
    // Required to allow the RayPickInteractor interaction

    simulation::BehaviorUpdatePositionVisitor updatePos(dt);
    context->execute(&updatePos);


    simulation::MechanicalBeginIntegrationVisitor beginVisitor(dt);
    context->execute(&beginVisitor);

    // Free Motion
    simulation::SolveVisitor freeMotion(dt, true);
    context->execute(&freeMotion);
    simulation::MechanicalPropagateFreePositionVisitor().execute(context);

    core::componentmodel::behavior::BaseMechanicalState::VecId dx_id = core::componentmodel::behavior::BaseMechanicalState::VecId::dx();
    simulation::MechanicalVOpVisitor(dx_id).execute( context);
    simulation::MechanicalPropagateDxVisitor(dx_id).execute( context);
    simulation::MechanicalVOpVisitor(dx_id).execute( context);

#ifdef DISPLAY_TIME
    sout<<" Free Motion " << ( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;

    time = (double) timer->getTime();
#endif

    // Collision detection and response creation
    computeCollision();

#ifdef DISPLAY_TIME
    sout<<" computeCollision " << ( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;
    time = (double) timer->getTime();
#endif
//	MechanicalResetContactForceVisitor().execute(context);

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }

    build_LCP();

#ifdef DISPLAY_TIME
    sout<<" build_LCP " << ( (double) timer->getTime() - time)*timeScale<<" ms" <<sendl;
    time = (double) timer->getTime();
#endif

    double _tol = tol.getValue();
    int _maxIt = maxIt.getValue();
    /*
    	if (_mu > 0.0)
    	{

    		lcp->setNbConst(_numConstraints);
    		lcp->setTol(_tol);
    		helper::nlcp_gaussseidel(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _mu, _tol, _maxIt, initial_guess.getValue());
    		if (this->f_printLog.getValue()) helper::afficheLCP(_dFree->ptr(), _W->lptr(), _result->ptr(),_numConstraints);
    	}
    	else
    	{

    //		helper::lcp_lexicolemke(_numConstraints, _dFree->ptr(), _W->lptr(), _A.lptr(), _result->ptr());
    		helper::gaussSeidelLCP1(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _tol, _maxIt);
    		if (this->f_printLog.getValue()) helper::afficheLCP(_dFree->ptr(), _W->lptr(), _result->ptr(),_numConstraints);
    	}
    */

    if (! initial_guess.getValue()) _f.clear();

#ifdef CHECK
    if (check_gpu.getValue())
    {
        real t1,t2;

        if (_mu > 0.0)
        {
            f_check.resize(_numConstraints,MBSIZE);
            for (unsigned i=0; i<_numConstraints; i++) f_check[i] = _f[i];
            t2 = sofa::gpu::cuda::CudaLCP<real>::CudaNlcp_gaussseidel(useGPU_d.getValue(),_numConstraints, _dFree.getCudaVector(), _W.getCudaMatrix(), f_check.getCudaVector(), _mu,_tol, _maxIt);

            t1 = sofa::gpu::cuda::CudaLCP<real>::CudaNlcp_gaussseidel(0,_numConstraints, _dFree.getCudaVector(), _W.getCudaMatrix(), _f.getCudaVector(), _mu,_tol, _maxIt);
        }
        else
        {
            f_check.resize(_numConstraints);
            for (unsigned i=0; i<_numConstraints; i++) f_check[i] = _f[i];
            t2 = sofa::gpu::cuda::CudaLCP<real>::CudaGaussSeidelLCP1(useGPU_d.getValue(),_numConstraints, _dFree.getCudaVector(), _W.getCudaMatrix(), f_check.getCudaVector(), _tol, _maxIt);

            t1 = sofa::gpu::cuda::CudaLCP<real>::CudaGaussSeidelLCP1(0,_numConstraints, _dFree.getCudaVector(), _W.getCudaMatrix(), _f.getCudaVector(), _tol, _maxIt);
        }

        if ((t1-t2>CHECK) || (t1-t2<-CHECK))
        {
            sout << "Error(" << useGPU_d.getValue() << ") dim(" << _numConstraints << ") : (cpu," << t1 << ") (gpu,(" << t2 << ")" << sendl;
        }

    }
    else
    {
#endif
        double error = 0.0;

        if (_mu > 0.0)
        {
            error = sofa::gpu::cuda::CudaLCP<real>::CudaNlcp_gaussseidel(useGPU_d.getValue(),_numConstraints, _dFree.getCudaVector(), _W.getCudaMatrix(), _f.getCudaVector(), _mu,(real)_tol, _maxIt);
        }
        else
        {
            error = sofa::gpu::cuda::CudaLCP<real>::CudaGaussSeidelLCP1(useGPU_d.getValue(),_numConstraints, _dFree.getCudaVector(), _W.getCudaMatrix(), _f.getCudaVector(), (real)_tol, _maxIt);
        }

        if (error > _tol) sout << "No convergence in gaussSeidelLCP1 : error = " << error << sendl;

#ifdef CHECK
    }
#endif

#ifdef DISPLAY_TIME
    sout<<" solve_LCP" <<( (double) timer->getTime() - time)*timeScale<<" ms" <<sendl;
    time = (double) timer->getTime();
#endif

    if (initial_guess.getValue())
        keepContactForcesValue();


//	MechanicalApplyContactForceVisitor(_result).execute(context);
    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->applyContactForce(&_f);
    }

    simulation::MechanicalPropagateAndAddDxVisitor().execute( context);
    simulation::MechanicalPropagatePositionAndVelocityVisitor().execute(context);


//	MechanicalResetContactForceVisitor().execute(context);
    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }
#ifdef DISPLAY_TIME
    sout<<" contactCorrections" <<( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;
    //sout << "<<<<<< End display MasterContactSolver time." << sendl;
#endif


    //switch lcp
    //sout << "switch lcp : " << lcp << endl;

    //lcp1.wait();
    //lcp1.lock();
    //lcp2.wait();
    //lcp2.lock();

    //	lcp = (lcp == &lcp1) ? &lcp2 : &lcp1;
    //	_A =&lcp->A;
    //	_W = &lcp->W;
    //	_dFree = &lcp->dFree;
    //	_result = &lcp->f;


    //lcp1.unlock();
    //lcp2.unlock();
    //sout << "new lcp : " << lcp << endl;

    //struct timespec ts;
    //ts.tv_sec = 0;
    //ts.tv_nsec = 60 *1000 *1000;
//	nanosleep(&ts, NULL);

    simulation::MechanicalEndIntegrationVisitor endVisitor(dt);
    context->execute(&endVisitor);
}

SOFA_DECL_CLASS(CudaMasterContactSolver)

int CudaMasterContactSolverClass = core::RegisterObject("Cuda Constraint solver")
        .add< CudaMasterContactSolver<float> >(true)
        .add< CudaMasterContactSolver<double> >()
        ;

} // namespace odesolver

} // namespace component

} // namespace sofa
