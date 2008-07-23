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
#ifndef SOFA_COMPONENT_ODESOLVER_CUDAMASTERCONTACTSOLVER_H
#define SOFA_COMPONENT_ODESOLVER_CUDAMASTERCONTACTSOLVER_H

#include <sofa/simulation/common/MasterSolverImpl.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/core/componentmodel/behavior/BaseConstraintCorrection.h>
#include <sofa/gpu/cuda/CudaLCP.h>
#include "CudaTypesBase.h"
#include <sofa/component/linearsolver/FullMatrix.h>

#define CHECK 0.01
//#define DISPLAY_TIME

namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::defaulttype;
using namespace sofa::component::linearsolver;
using namespace helper::system::thread;
using namespace sofa::gpu::cuda;

class CudaMechanicalGetConstraintValueVisitor : public simulation::MechanicalVisitor
{
public:
    CudaMechanicalGetConstraintValueVisitor(defaulttype::BaseVector * v): _v(v) {}

    virtual Result fwdConstraint(simulation::Node*,core::componentmodel::behavior::BaseConstraint* c)
    {
        c->getConstraintValue(_v);
        return RESULT_CONTINUE;
    }
private:
    defaulttype::BaseVector * _v;
};

template<class real>
class MechanicalGetConstraintValueVisitor : public simulation::MechanicalVisitor
{
public:

    MechanicalGetConstraintValueVisitor(defaulttype::BaseVector * v)
    {
        real * data = ((CudaBaseVector<real> *) v)->getCudaVector().hostWrite();
        _v = new FullVector<real>(data,0);
    }

    virtual Result fwdConstraint(simulation::Node*,core::componentmodel::behavior::BaseConstraint* c)
    {
        c->getConstraintValue(_v);
        return RESULT_CONTINUE;
    }
private:
    FullVector<real> * _v;
};



class CudaMechanicalGetContactIDVisitor : public simulation::MechanicalVisitor
{
public:
    CudaMechanicalGetContactIDVisitor(long *id, unsigned int offset = 0)
        : _id(id),_offset(offset) {}

    virtual Result fwdConstraint(simulation::Node*,core::componentmodel::behavior::BaseConstraint* c)
    {
        c->getConstraintId(_id, _offset);
        return RESULT_CONTINUE;
    }

private:
    long *_id;
    unsigned int _offset;
};

template<class real>
class CudaMasterContactSolver : public sofa::simulation::MasterSolverImpl, public virtual sofa::core::objectmodel::BaseObject
{
public:
    typedef real Real;
    Data<bool> initial_guess_d;
#ifdef CHECK
    Data<bool> check_gpu;
#endif
    Data <double> tol_d;
    Data <int> maxIt_d;
    Data <double> mu_d;

    Data<int> useGPU_d;

    CudaMasterContactSolver();

    void step (double dt);

    virtual void init();

private:
    std::vector<core::componentmodel::behavior::BaseConstraintCorrection*> constraintCorrections;
    void computeInitialGuess();
    void keepContactForcesValue();

    void build_LCP();

    CudaBaseMatrix<real> _W;
    CudaBaseVector<real> _dFree, _f;

#ifdef CHECK
    CudaBaseVector<real> f_check;
#endif

    unsigned int _numConstraints;
    double _mu;
    simulation::tree::GNode *context;

    typedef struct
    {
        Vector3 n;
        Vector3 t;
        Vector3 s;
        Vector3 F;
        long id;

    } contactBuf;

    contactBuf *_PreviousContactList;
    unsigned int _numPreviousContact;
    long *_cont_id_list;
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
