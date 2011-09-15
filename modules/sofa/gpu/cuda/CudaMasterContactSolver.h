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
#ifndef SOFA_COMPONENT_ODESOLVER_CUDAMASTERCONTACTSOLVER_H
#define SOFA_COMPONENT_ODESOLVER_CUDAMASTERCONTACTSOLVER_H

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/simulation/common/CollisionAnimationLoop.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/core/behavior/BaseConstraintCorrection.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/component/odesolver/OdeSolverImpl.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/helper/set.h>
#include <sofa/gpu/cuda/CudaLCP.h>
#include "CudaTypesBase.h"

//#define CHECK 0.01
#define DISPLAY_TIME

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

class MechanicalResetContactForceVisitor : public simulation::BaseMechanicalVisitor
{
public:
    //core::MultiVecDerivId force;
    MechanicalResetContactForceVisitor(const core::ExecParams* params)
        : simulation::BaseMechanicalVisitor(params)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms)
    {
        ms->resetContactForce();
        return RESULT_CONTINUE;
    }

    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms)
    {
        ms->resetForce();
        return RESULT_CONTINUE;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
};

/* ACTION 2 : Apply the Contact Forces on mechanical models & Compute displacements */
class MechanicalApplyContactForceVisitor : public simulation::BaseMechanicalVisitor
{
public:
    //core::MultiVecDerivId force;
    MechanicalApplyContactForceVisitor(const core::ExecParams* params /* PARAMS FIRST */, double *f)
        : simulation::BaseMechanicalVisitor(params)
        ,_f(f)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }
    virtual Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms)
    {
        ms->applyContactForce(_f);
        return RESULT_CONTINUE;
    }

    virtual Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms)
    {
        ms->applyContactForce(_f);
        return RESULT_CONTINUE;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif

private:
    double *_f; // vector of contact forces from lcp //
    // to be multiplied by constraint direction in mechanical models //

};

/* ACTION 3 : gets the vector of constraint values */
/* ACTION 3 : gets the vector of constraint values */
template<class real>
class MechanicalGetConstraintValueVisitor : public simulation::BaseMechanicalVisitor
{
public:

    MechanicalGetConstraintValueVisitor(const core::ConstraintParams* params /* PARAMS FIRST */, BaseVector * v)
        : simulation::BaseMechanicalVisitor(params)
        , cparams(params)
    {
        real * data = ((CudaBaseVector<real> *) v)->getCudaVector().hostWrite();
        _v = new FullVector<real>(data,0);
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual Result fwdConstraintSet(simulation::Node* /*node*/, core::behavior::BaseConstraintSet* cSet)
    {
        if (core::behavior::BaseConstraint *c=dynamic_cast<core::behavior::BaseConstraint*>(cSet))
        {
            //sout << c->getName()<<"->getConstraintValue()"<<sendl;
            c->getConstraintValue(_v /*, _numContacts*/);
        }
        return RESULT_CONTINUE;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
private:
    FullVector<real> * _v; // vector for constraint values
    // unsigned int &_numContacts; // we need an offset to fill the vector _v if differents contact class are created
};

class CudaMechanicalGetConstraintValueVisitor  : public simulation::BaseMechanicalVisitor
{
public:
    CudaMechanicalGetConstraintValueVisitor(const core::ConstraintParams* params /* PARAMS FIRST */, defaulttype::BaseVector * v)
        : simulation::BaseMechanicalVisitor(params)
        , cparams(params)
        , _v(v)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual Result fwdConstraintSet(simulation::Node*,core::behavior::BaseConstraintSet* cSet)
    {

        if (core::behavior::BaseConstraint *c=dynamic_cast<core::behavior::BaseConstraint*>(cSet))
        {
            c->getConstraintViolation(cparams /* PARAMS FIRST */, _v);
        }
        return RESULT_CONTINUE;
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
private:
    /// Constraint parameters
    const sofa::core::ConstraintParams *cparams;

    defaulttype::BaseVector * _v;
};

class MechanicalGetConstraintInfoVisitor : public simulation::BaseMechanicalVisitor
{
public:
    typedef core::behavior::BaseConstraint::VecConstraintBlockInfo VecConstraintBlockInfo;
    typedef core::behavior::BaseConstraint::VecPersistentID VecPersistentID;
    typedef core::behavior::BaseConstraint::VecConstCoord VecConstCoord;
    typedef core::behavior::BaseConstraint::VecConstDeriv VecConstDeriv;
    typedef core::behavior::BaseConstraint::VecConstArea VecConstArea;

    MechanicalGetConstraintInfoVisitor(const core::ExecParams* params /* PARAMS FIRST */, VecConstraintBlockInfo& blocks, VecPersistentID& ids, VecConstCoord& positions, VecConstDeriv& directions, VecConstArea& areas)
        : simulation::BaseMechanicalVisitor(params), _blocks(blocks), _ids(ids), _positions(positions), _directions(directions), _areas(areas)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual Result fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet)
    {

        if (core::behavior::BaseConstraint *c=dynamic_cast<core::behavior::BaseConstraint*>(cSet))
        {
            ctime_t t0 = begin(node, c);
            c->getConstraintInfo(_blocks, _ids, _positions, _directions, _areas);
            end(node, c, t0);
        }
        return RESULT_CONTINUE;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
private:
    VecConstraintBlockInfo& _blocks;
    VecPersistentID& _ids;
    VecConstCoord& _positions;
    VecConstDeriv& _directions;
    VecConstArea& _areas;
};

template<class real>
class CudaMasterContactSolver : public sofa::simulation::CollisionAnimationLoop
{
public:
    SOFA_CLASS(CudaMasterContactSolver,sofa::simulation::CollisionAnimationLoop);
    Data<int> useGPU_d;
#ifdef DISPLAY_TIME
    Data<bool> print_info;
#endif

    Data<bool> initial_guess;
    Data < double > tol;
    Data < int > maxIt;
    Data < double > mu;

    Data < helper::set<int> > constraintGroups;

    CudaMasterContactSolver();
    // virtual const char* getTypeName() const { return "AnimationLoop"; }

    virtual void step (const core::ExecParams* params /* PARAMS FIRST  = core::ExecParams::defaultInstance()*/, double dt);

    //virtual void propagatePositionAndVelocity(double t, VecId x, VecId v);

    virtual void init();
    //LCP* getLCP(void) {return (lcp == &lcp1) ? &lcp2 : &lcp1;};

private:
    std::vector<core::behavior::BaseConstraintCorrection*> constraintCorrections;
    void computeInitialGuess();
    void keepContactForcesValue();

    void build_LCP();

    unsigned int _numConstraints;
    unsigned int _realNumConstraints;
    double _mu;
    simulation::Node *context;

    CudaBaseMatrix<real> _W;
    CudaBaseVector<real> _dFree, _f;
#ifdef CHECK
    CudaBaseVector<real> f_check;
#endif


    typedef core::behavior::BaseConstraint::ConstraintBlockInfo ConstraintBlockInfo;
    typedef core::behavior::BaseConstraint::PersistentID PersistentID;
    typedef core::behavior::BaseConstraint::ConstCoord ConstCoord;
    typedef core::behavior::BaseConstraint::ConstDeriv ConstDeriv;
    typedef core::behavior::BaseConstraint::ConstArea ConstArea;

    typedef core::behavior::BaseConstraint::VecConstraintBlockInfo VecConstraintBlockInfo;
    typedef core::behavior::BaseConstraint::VecPersistentID VecPersistentID;
    typedef core::behavior::BaseConstraint::VecConstCoord VecConstCoord;
    typedef core::behavior::BaseConstraint::VecConstDeriv VecConstDeriv;
    typedef core::behavior::BaseConstraint::VecConstArea VecConstArea;

    class ConstraintBlockBuf
    {
    public:
        std::map<PersistentID,int> persistentToConstraintIdMap;
        int nbLines; ///< how many dofs (i.e. lines in the matrix) are used by each constraint
    };

    std::map<core::behavior::BaseConstraint*, ConstraintBlockBuf> _previousConstraints;
    helper::vector< double > _previousForces;

    VecConstraintBlockInfo _constraintBlockInfo;
    VecPersistentID _constraintIds;
    VecConstCoord _constraintPositions;
    VecConstDeriv _constraintDirections;
    VecConstArea _constraintAreas;

    helper::vector<unsigned> constraintRenumbering,constraintReinitialize;
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
