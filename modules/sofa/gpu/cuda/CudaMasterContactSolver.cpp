/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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

#include <sofa/helper/LCPcalc.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/SolveVisitor.h>

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

static unsigned MAX_NUM_CONSTRAINTS=2048;

template<class real>
CudaMasterContactSolver<real>::CudaMasterContactSolver()
    :
    initial_guess_d(initData(&initial_guess_d, true, "initial_guess","activate LCP results history to improve its resolution performances."))
#ifdef CHECK
    ,check_gpu(initData(&check_gpu, true, "checkGPU", "verification of lcp error"))
#endif
    ,tol_d( initData(&tol_d, 0.001, "tolerance", "tolerance"))
    ,maxIt_d(initData(&maxIt_d, 100, "maxIt", "iterations of gauss seidel"))
    ,mu_d( initData(&mu_d, 0.6, "mu", ""))
    ,useGPU_d(initData(&useGPU_d,8, "useGPU", "compute LCP using GPU"))
{

    _W.resize(MAX_NUM_CONSTRAINTS,MAX_NUM_CONSTRAINTS);
    _dFree.resize(MAX_NUM_CONSTRAINTS);
    _f.resize(MAX_NUM_CONSTRAINTS);
    _numConstraints = 0;
    _mu = mu_d.getValue();

    _numPreviousContact=0;
    _PreviousContactList = (contactBuf *)malloc(MAX_NUM_CONSTRAINTS * sizeof(contactBuf));
    _cont_id_list = (long *)malloc(MAX_NUM_CONSTRAINTS * sizeof(long));
}

template<class real>
void CudaMasterContactSolver<real>::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    context->get<core::componentmodel::behavior::BaseConstraintCorrection>(&constraintCorrections, core::objectmodel::BaseContext::SearchDown);
}

template<class real>
void CudaMasterContactSolver<real>::build_LCP()
{
    _numConstraints = 0;

    simulation::MechanicalResetConstraintVisitor().execute(context);
    simulation::MechanicalAccumulateConstraint(_numConstraints, _mu).execute(context);

    if (_numConstraints > MAX_NUM_CONSTRAINTS)
    {
        cerr<<endl<<"Warning in MasterContactSolver, maximum number of contacts exceeded, "<< _numConstraints <<" contacts detected"<<endl;
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

    if (useGPU_d.getValue())
    {
        CudaMechanicalGetConstraintValueVisitor(&_dFree).execute(context);

        for (unsigned int i=0; i<constraintCorrections.size(); i++)
        {
            core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
            cc->getCompliance(&_W);
        }
    }
    else
    {
        MechanicalGetConstraintValueVisitor<real>(&_dFree).execute(context);

        for (unsigned int i=0; i<constraintCorrections.size(); i++)
        {
            core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
            real * data = _W.getCudaMatrix().hostWrite();

            FullMatrix<real> * w = new FullMatrix<real>(data,_W.colSize(),_W.rowSize());

            cc->getCompliance(w);
        }
    }

    if (initial_guess_d.getValue())
    {
        CudaMechanicalGetContactIDVisitor(_cont_id_list).execute(context);
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

template<class real>
void CudaMasterContactSolver<real>::keepContactForcesValue()
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

template<class real>
void CudaMasterContactSolver<real>::step(double dt)
{
    _mu = mu_d.getValue();

    real _tol = (real) tol_d.getValue();
    int _maxIt = maxIt_d.getValue();

    context = dynamic_cast<simulation::tree::GNode *>(this->getContext()); // access to current node

#ifdef DISPLAY_TIME
    CTime *timer = new CTime();
    double time = 0;
    time = (double) timer->getTime();
    std::cout<<" ********* Start Iteration *********" <<std::endl;
#endif

    simulation::BehaviorUpdatePositionVisitor updatePos(dt);
    context->execute(&updatePos);

    simulation::MechanicalBeginIntegrationVisitor beginVisitor(dt);
    context->execute(&beginVisitor);


    // Free Motion
    simulation::SolveVisitor freeMotion(dt);
    context->execute(&freeMotion);

    simulation::MechanicalPropagateFreePositionVisitor().execute(context);

    core::componentmodel::behavior::BaseMechanicalState::VecId dx_id = core::componentmodel::behavior::BaseMechanicalState::VecId::dx();
    simulation::MechanicalVOpVisitor(dx_id).execute( context);
    simulation::MechanicalPropagateDxVisitor(dx_id).execute( context);
    simulation::MechanicalVOpVisitor(dx_id).execute( context);

#ifdef DISPLAY_TIME
    std::cout<<" Free Motion :        " << ( (double) timer->getTime() - time)*0.001 <<" ms" <<std::endl;
    time = (double) timer->getTime();
#endif

    computeCollision();

#ifdef DISPLAY_TIME
    std::cout<<" computeCollision :  " << ( (double) timer->getTime() - time)*0.001 <<" ms" <<std::endl;
    time = (double) timer->getTime();
#endif

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }

    build_LCP();

#ifdef DISPLAY_TIME
    std::cout<<" build_LCP :         " << ( (double) timer->getTime() - time)*0.001<<" ms" <<std::endl;
    time = (double) timer->getTime();
#endif

    if (! initial_guess_d.getValue()) _f.clear();

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
            std::cout << "Error(" << useGPU_d.getValue() << ") dim(" << _numConstraints << ") : (cpu," << t1 << ") (gpu,(" << t2 << ")" << std::endl;
        }
    }
    else
    {
#endif
        real error;

        if (_mu > 0.0)
        {
            error = sofa::gpu::cuda::CudaLCP<real>::CudaNlcp_gaussseidel(useGPU_d.getValue(),_numConstraints, _dFree.getCudaVector(), _W.getCudaMatrix(), _f.getCudaVector(), _mu,_tol, _maxIt);
        }
        else
        {
            error = sofa::gpu::cuda::CudaLCP<real>::CudaGaussSeidelLCP1(useGPU_d.getValue(),_numConstraints, _dFree.getCudaVector(), _W.getCudaMatrix(), _f.getCudaVector(), _tol, _maxIt);
        }

        if (error > _tol) std::cout << "No convergence in gaussSeidelLCP1 : error = " << error << std::endl;

#ifdef CHECK
    }
#endif

#ifdef DISPLAY_TIME
    std::cout<<" solve_LCP :         " <<( (double) timer->getTime() - time)*0.001<<" ms" <<std::endl;
    time = (double) timer->getTime();
#endif

    if (initial_guess_d.getValue()) keepContactForcesValue();

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->applyContactForce(&_f);
    }

    simulation::MechanicalPropagateAndAddDxVisitor().execute( context);

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }

#ifdef DISPLAY_TIME
    std::cout<<" contactCorrections : " <<( (double) timer->getTime() - time)*0.001 <<" ms" <<std::endl;
#endif

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
