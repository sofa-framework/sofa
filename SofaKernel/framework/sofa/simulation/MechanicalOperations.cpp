/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/MechanicalMatrixVisitor.h>
#include <sofa/simulation/MechanicalComputeEnergyVisitor.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/VecId.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/defaulttype/BaseMatrix.h>

using namespace sofa::core;
namespace sofa
{

namespace simulation
{

namespace common
{

MechanicalOperations::MechanicalOperations(const sofa::core::MechanicalParams* mparams, sofa::core::objectmodel::BaseContext* ctx, bool precomputedTraversalOrder)
    :mparams(*mparams),ctx(ctx),executeVisitor(*ctx,precomputedTraversalOrder)
{
}

MechanicalOperations::MechanicalOperations(const sofa::core::ExecParams* params, sofa::core::objectmodel::BaseContext* ctx, bool precomputedTraversalOrder)
    :mparams(*params),ctx(ctx),executeVisitor(*ctx,precomputedTraversalOrder)
{
}

void MechanicalOperations::setX(core::MultiVecCoordId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::VecCoordId::position());
    mparams.setX(v);
}

void MechanicalOperations::setX(core::ConstMultiVecCoordId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::VecCoordId::position());
    mparams.setX(v);
}

void MechanicalOperations::setV(core::MultiVecDerivId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::VecDerivId::velocity());
    mparams.setV(v);
}

void MechanicalOperations::setV(core::ConstMultiVecDerivId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::VecDerivId::velocity());
    mparams.setV(v);
}

void MechanicalOperations::setF(core::MultiVecDerivId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::VecDerivId::force());
    mparams.setF(v);
}

void MechanicalOperations::setF(core::ConstMultiVecDerivId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::VecDerivId::force());
    mparams.setF(v);
}

void MechanicalOperations::setDx(core::MultiVecDerivId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::VecDerivId::dx());
    mparams.setDx(v);
}

void MechanicalOperations::setDx(core::ConstMultiVecDerivId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::VecDerivId::dx());
    mparams.setDx(v);
}

void MechanicalOperations::setDf(core::MultiVecDerivId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::VecDerivId::dforce());
    mparams.setDf(v);
}

void MechanicalOperations::setDf(core::ConstMultiVecDerivId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::VecDerivId::dforce());
    mparams.setDf(v);
}


/// Propagate the given displacement through all mappings
void MechanicalOperations::propagateDx(core::MultiVecDerivId dx, bool ignore_flag)
{
    setDx(dx);
    executeVisitor( MechanicalPropagateDxVisitor(&mparams, dx, false, ignore_flag) );
}

/// Propagate the given displacement through all mappings and reset the current force delta
void MechanicalOperations::propagateDxAndResetDf(core::MultiVecDerivId dx, core::MultiVecDerivId df)
{
    setDx(dx);
    setDf(df);
    executeVisitor( MechanicalPropagateDxAndResetForceVisitor(&mparams, dx, df, false) );
}

/// Propagate the given position through all mappings
void MechanicalOperations::propagateX(core::MultiVecCoordId x)
{
    setX(x);
    MechanicalPropagateOnlyPositionVisitor visitor(&mparams, 0.0, x, false); //Don't ignore the masks
    executeVisitor( visitor );
}

/// Propagate the given velocity through all mappings
void MechanicalOperations::propagateV(core::MultiVecDerivId v)
{
    setV(v);
    MechanicalPropagateOnlyVelocityVisitor visitor(&mparams, 0.0, v, false); //Don't ignore the masks
    executeVisitor( visitor );
}

/// Propagate the given position and velocity through all mappings
void MechanicalOperations::propagateXAndV(core::MultiVecCoordId x, core::MultiVecDerivId v)
{
    setX(x);
    setV(v);
    MechanicalPropagateOnlyPositionAndVelocityVisitor visitor(&mparams, 0.0, x, v, false); //Don't ignore the masks
    executeVisitor( visitor );
}

/// Propagate the given position through all mappings and reset the current force delta
void MechanicalOperations::propagateXAndResetF(core::MultiVecCoordId x, core::MultiVecDerivId f)
{
    setX(x);
    setF(f);
    MechanicalPropagateOnlyPositionAndResetForceVisitor visitor(&mparams, x, f, false);
    executeVisitor( visitor );
}

/// Apply projective constraints to the given position vector
void MechanicalOperations::projectPosition(core::MultiVecCoordId x, SReal time)
{
    setX(x);
    executeVisitor( MechanicalProjectPositionVisitor(&mparams, time, x) );
}

/// Apply projective constraints to the given velocity vector
void MechanicalOperations::computeEnergy(SReal &kineticEnergy, SReal &potentialEnergy)
{
    kineticEnergy=0;
    potentialEnergy=0;
    sofa::simulation::MechanicalComputeEnergyVisitor *energyVisitor = new sofa::simulation::MechanicalComputeEnergyVisitor(&mparams);
    executeVisitor(energyVisitor);
    kineticEnergy=energyVisitor->getKineticEnergy();
    potentialEnergy=energyVisitor->getPotentialEnergy();
}
/// Apply projective constraints to the given velocity vector
void MechanicalOperations::projectVelocity(core::MultiVecDerivId v, SReal time)
{
    setV(v);
    executeVisitor( MechanicalProjectVelocityVisitor(&mparams, time, v) );
}

/// Apply projective constraints to the given vector
void MechanicalOperations::projectResponse(core::MultiVecDerivId dx, double **W)
{
    setDx(dx);
    executeVisitor( MechanicalApplyConstraintsVisitor(&mparams, dx, W) );
}

/// Apply projective constraints to the given position and velocity vectors
void MechanicalOperations::projectPositionAndVelocity(core::MultiVecCoordId x, core::MultiVecDerivId v, double time)
{
    setX(x);
    setV(v);
    executeVisitor( MechanicalProjectPositionAndVelocityVisitor(&mparams, time, x, v) );
}

void MechanicalOperations::addMdx(core::MultiVecDerivId res, core::MultiVecDerivId dx, SReal factor)
{
    setDx(dx);
    executeVisitor( MechanicalAddMDxVisitor(&mparams, res,dx,factor) );
}

///< res += factor M.dx
void MechanicalOperations::integrateVelocity(core::MultiVecDerivId res, core::ConstMultiVecCoordId x, core::ConstMultiVecDerivId v, SReal dt)
{
    executeVisitor( MechanicalVOpVisitor(&mparams, res,x,v,dt) );
}

///< res = x + v.dt
void MechanicalOperations::accFromF(core::MultiVecDerivId a, core::ConstMultiVecDerivId f) ///< a = M^-1 . f
{
    setDx(a);
    setF(f);

    executeVisitor( MechanicalAccFromFVisitor(&mparams, a) );
}

/// Compute the current force (given the latest propagated position and velocity)
void MechanicalOperations::computeForce(core::MultiVecDerivId result, bool clear, bool accumulate, bool neglectingCompliance)
{
    setF(result);
    if (clear)
    {
        executeVisitor( MechanicalResetForceVisitor(&mparams, result, false) );
        //finish();
    }
    executeVisitor( MechanicalComputeForceVisitor(&mparams, result, accumulate, neglectingCompliance) );
}


/// Compute the current force delta (given the latest propagated displacement)
void MechanicalOperations::computeDf(core::MultiVecDerivId df, bool clear, bool accumulate)
{
    setDf(df);
    if (clear)
    {
        executeVisitor( MechanicalResetForceVisitor(&mparams, df, false) );
        //	finish();
    }
    executeVisitor( MechanicalComputeDfVisitor( &mparams, df,  accumulate) );
}

/// Compute the current force delta (given the latest propagated velocity)
void MechanicalOperations::computeDfV(core::MultiVecDerivId df, bool clear, bool accumulate)
{
    core::ConstMultiVecDerivId dx = mparams.dx();
    mparams.setDx(mparams.v());
    setDf(df);
    if (clear)
    {
        executeVisitor( MechanicalResetForceVisitor(&mparams, df, false) );
        //finish();
    }
    executeVisitor( MechanicalComputeDfVisitor(&mparams, df, accumulate) );
    mparams.setDx(dx);
}

/// accumulate $ df += (m M + b B + k K) dx $ (given the latest propagated displacement)
void MechanicalOperations::addMBKdx(core::MultiVecDerivId df, SReal m, SReal b, SReal k, bool clear, bool accumulate)
{
    setDf(df);
    if (clear)
    {
        executeVisitor( MechanicalResetForceVisitor(&mparams, df, true) );
        //finish();
    }
    mparams.setBFactor(b);
    mparams.setKFactor(k);
    mparams.setMFactor(m);
    executeVisitor( MechanicalAddMBKdxVisitor(&mparams, df, accumulate) );
}

/// accumulate $ df += (m M + b B + k K) velocity $
void MechanicalOperations::addMBKv(core::MultiVecDerivId df, SReal m, SReal b, SReal k, bool clear, bool accumulate)
{
    core::ConstMultiVecDerivId dx = mparams.dx();
    mparams.setDx(mparams.v());
    setDf(df);
    if (clear)
    {
        executeVisitor( MechanicalResetForceVisitor(&mparams, df, true) );
        //finish();
    }
    mparams.setBFactor(b);
    mparams.setKFactor(k);
    mparams.setMFactor(m);
    /* useV = true */
    executeVisitor( MechanicalAddMBKdxVisitor(&mparams, df, accumulate) );
    mparams.setDx(dx);
}



/// Add dt*Gravity to the velocity
void MechanicalOperations::addSeparateGravity(SReal dt, core::MultiVecDerivId result)
{
    mparams.setDt(dt);
    setV(result);
    executeVisitor( MechanicalAddSeparateGravityVisitor(&mparams, result) );
}

void MechanicalOperations::computeContactForce(core::MultiVecDerivId result)
{
    setF(result);
    executeVisitor( MechanicalResetForceVisitor(&mparams, result, false) );
    //finish();
    executeVisitor( MechanicalComputeContactForceVisitor(&mparams, result) );
}

void MechanicalOperations::computeContactDf(core::MultiVecDerivId df)
{
    setDf(df);
    executeVisitor( MechanicalResetForceVisitor(&mparams, df, false) );
    //finish();
}

void MechanicalOperations::computeAcc(SReal t, core::MultiVecDerivId a, core::MultiVecCoordId x, core::MultiVecDerivId v)
{
    MultiVecDerivId f( VecDerivId::force() );
    setF(f);
    setDx(a);
    setX(x);
    setV(v);
    executeVisitor( MechanicalProjectPositionAndVelocityVisitor(&mparams, t,x,v) );
    executeVisitor( MechanicalPropagateOnlyPositionAndVelocityVisitor(&mparams, t,x,v,
#ifdef SOFA_SUPPORT_MAPPED_MASS
            a,
#endif
            true) );
    computeForce(f);

    accFromF(a,f);
    projectResponse(a);
}

void MechanicalOperations::computeForce(SReal t, core::MultiVecDerivId f, core::MultiVecCoordId x, core::MultiVecDerivId v, bool neglectingCompliance)
{
    setF(f);
    setX(x);
    setV(v);
    executeVisitor( MechanicalProjectPositionAndVelocityVisitor(&mparams, t,x,v) );
    executeVisitor( MechanicalPropagateOnlyPositionAndVelocityVisitor(&mparams, t,x,v,
#ifdef SOFA_SUPPORT_MAPPED_MASS
            a,
#endif
            true) );
    computeForce(f,true,true,neglectingCompliance);

    projectResponse(f);
}

void MechanicalOperations::computeContactAcc(SReal t, core::MultiVecDerivId a, core::MultiVecCoordId x, core::MultiVecDerivId v)
{
    MultiVecDerivId f( VecDerivId::force() );
    setF(f);
    setDx(a);
    setX(x);
    setV(v);
    executeVisitor( MechanicalProjectPositionAndVelocityVisitor(&mparams, t,x,v) );
    executeVisitor( MechanicalPropagateOnlyPositionAndVelocityVisitor(&mparams, t,x,v,
#ifdef SOFA_SUPPORT_MAPPED_MASS
            a,
#endif
            true) );
    computeContactForce(f);

    accFromF(a,f);
    projectResponse(a);
}





using sofa::core::behavior::LinearSolver;
using sofa::core::objectmodel::BaseContext;

/*
void MechanicalOperations::solveConstraint(SReal dt, MultiVecDerivId id, core::ConstraintParams::ConstOrder order )
{
  core::ConstraintParams cparams(mparams    // PARAMS FIRST //, order);
  mparams.setDt(dt);
  assert( order == core::ConstraintParams::VEL || order == core::ConstraintParams::ACC);
  cparams.setV( id);
  solveConstraint(&cparams    // PARAMS FIRST //, id);
}

void MechanicalOperations::solveConstraint(SReal dt, MultiVecCoordId id, core::ConstraintParams::ConstOrder order)
{
  core::ConstraintParams cparams(mparams    // PARAMS FIRST //, order);
  mparams.setDt(dt);
  assert( order == core::ConstraintParams::POS);
  cparams.setX( id);
  solveConstraint(&cparams    // PARAMS FIRST //, id);
}
*/

void MechanicalOperations::solveConstraint(MultiVecId id, core::ConstraintParams::ConstOrder order)
{
    cparams.setOrder(order);

//	ctx->serr << "MechanicalOperations::solveConstraint" << ctx->sendl;
    helper::vector< core::behavior::ConstraintSolver* > constraintSolverList;

    ctx->get<core::behavior::ConstraintSolver>(&constraintSolverList, ctx->getTags(), BaseContext::Local);
    if (constraintSolverList.empty())
    {
        //ctx->sout << "No ConstraintSolver found." << ctx->sendl;
        return;
    }

//	ctx->serr << "MechanicalOperations::solveConstraint found solvers" << ctx->sendl;
    for (helper::vector< core::behavior::ConstraintSolver* >::iterator it=constraintSolverList.begin(); it!=constraintSolverList.end(); ++it)
    {
        (*it)->solveConstraint(&cparams, id);
    }
}

void MechanicalOperations::m_resetSystem()
{
    LinearSolver* s = ctx->get<LinearSolver>(ctx->getTags(), BaseContext::SearchDown);
    if (!s)
    {
        ctx->serr << "ERROR: requires a LinearSolver."<<ctx->sendl;
        return;
    }
    s->resetSystem();
}

void MechanicalOperations::m_setSystemMBKMatrix(SReal mFact, SReal bFact, SReal kFact)
{
    LinearSolver* s = ctx->get<LinearSolver>(ctx->getTags(), BaseContext::SearchDown);
    if (!s)
    {
        ctx->serr << "ERROR:  requires a LinearSolver."<<ctx->sendl;
        return;
    }
    mparams.setMFactor(mFact);
    mparams.setBFactor(bFact);
    mparams.setKFactor(kFact);
    s->setSystemMBKMatrix(&mparams);
}

void MechanicalOperations::m_setSystemRHVector(core::MultiVecDerivId v)
{
    LinearSolver* s = ctx->get<LinearSolver>(ctx->getTags(), BaseContext::SearchDown);
    if (!s)
    {
        ctx->serr << "ERROR:  requires a LinearSolver."<<ctx->sendl;
        return;
    }
    s->setSystemRHVector(v);
}

void MechanicalOperations::m_setSystemLHVector(core::MultiVecDerivId v)
{
    LinearSolver* s = ctx->get<LinearSolver>(ctx->getTags(), BaseContext::SearchDown);
    if (!s)
    {
        ctx->serr << "ERROR:  requires a LinearSolver."<<ctx->sendl;
        return;
    }
    s->setSystemLHVector(v);

}

void MechanicalOperations::m_solveSystem()
{
    LinearSolver* s = ctx->get<LinearSolver>(ctx->getTags(), BaseContext::SearchDown);
    if (!s)
    {
        ctx->serr << "ERROR:  requires a LinearSolver."<<ctx->sendl;
        return;
    }
    s->solveSystem();
}

void MechanicalOperations::m_print( std::ostream& out )
{
    LinearSolver* s = ctx->get<LinearSolver>(ctx->getTags(), BaseContext::SearchDown);
    if (!s)
    {
        ctx->serr << "ERROR: requires a LinearSolver."<<ctx->sendl;
        return;
    }
    defaulttype::BaseMatrix* m = s->getSystemBaseMatrix();
    if (!m) return;
    //out << *m;
    int ny = m->rowSize();
    int nx = m->colSize();
    out << "[";
    for (int y=0; y<ny; ++y)
    {
        out << "[";
        for (int x=0; x<nx; x++)
            out << ' ' << m->element(x,y);
        out << "]";
    }
    out << "]";
}


// BaseMatrix & BaseVector Computations
void MechanicalOperations::getMatrixDimension(unsigned int *  const nbRow, unsigned int * const nbCol, sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    executeVisitor( MechanicalGetMatrixDimensionVisitor(&mparams, nbRow, nbCol, matrix) );
}

void MechanicalOperations::addMBK_ToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, SReal mFact, SReal bFact, SReal kFact)
{
    mparams.setMFactor(mFact);
    mparams.setBFactor(bFact);
    mparams.setKFactor(kFact);
    if (matrix != NULL)
    {
        executeVisitor( MechanicalAddMBK_ToMatrixVisitor(&mparams, matrix) );
    }
}

void MechanicalOperations::addSubMBK_ToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix,const helper::vector<unsigned> & subMatrixIndex, SReal mFact, SReal bFact, SReal kFact)
{
    mparams.setMFactor(mFact);
    mparams.setBFactor(bFact);
    mparams.setKFactor(kFact);
    if (matrix != NULL)
    {
        executeVisitor( MechanicalAddSubMBK_ToMatrixVisitor(&mparams, matrix, subMatrixIndex) );
    }
}



void MechanicalOperations::multiVector2BaseVector(core::ConstMultiVecId src, defaulttype::BaseVector *dest, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    if (dest != NULL)
    {
        executeVisitor( MechanicalMultiVectorToBaseVectorVisitor(&mparams, src, dest, matrix) );
    }
}


void MechanicalOperations::multiVectorPeqBaseVector(core::MultiVecDerivId dest, defaulttype::BaseVector *src, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    if (src != NULL)
    {
        executeVisitor( MechanicalMultiVectorPeqBaseVectorVisitor(&mparams, dest, src, matrix) );
    }
}




/// Dump the content of the given vector.
void MechanicalOperations::print( core::ConstMultiVecId /*v*/, std::ostream& /*out*/ )
{
}


void MechanicalOperations::printWithElapsedTime( core::ConstMultiVecId /*v*/, unsigned /*time*/, std::ostream& /*out*/ )
{
}

}

}

}
