/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/simulation/mechanicalvisitor/MechanicalMultiVectorToBaseVectorVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalMultiVectorFromBaseVectorVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalMultiVectorPeqBaseVectorVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalComputeEnergyVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateDxVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateDxAndResetForceVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyVelocityVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndResetForceVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalProjectPositionVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalProjectVelocityVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalApplyConstraintsVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalProjectPositionAndVelocityVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalAddMDxVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalVOpVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalAccFromFVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalResetForceVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalComputeForceVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalComputeDfVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalAddMBKdxVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalAddSeparateGravityVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalComputeContactForceVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalGetMatrixDimensionVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalAddMBK_ToMatrixVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalApplyProjectiveConstraint_ToMatrixVisitor.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/VecId.h>

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/core/behavior/ConstraintSolver.h>

#include <sofa/core/ObjectFactory.h>

#include <numeric>

using namespace sofa::core;

namespace sofa::simulation::common
{

using namespace sofa::simulation::mechanicalvisitor;

std::map<core::objectmodel::BaseContext*, bool> MechanicalOperations::hasShownMissingLinearSolverMap;

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
    if (v.getDefaultId().isNull()) v.setDefaultId(core::vec_id::write_access::position);
    mparams.setX(v);
}

void MechanicalOperations::setX(core::ConstMultiVecCoordId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::vec_id::write_access::position);
    mparams.setX(v);
}

void MechanicalOperations::setV(core::MultiVecDerivId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::vec_id::write_access::velocity);
    mparams.setV(v);
}

void MechanicalOperations::setV(core::ConstMultiVecDerivId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::vec_id::write_access::velocity);
    mparams.setV(v);
}

void MechanicalOperations::setF(core::MultiVecDerivId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::vec_id::write_access::force);
    mparams.setF(v);
}

void MechanicalOperations::setF(core::ConstMultiVecDerivId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::vec_id::write_access::force);
    mparams.setF(v);
}

void MechanicalOperations::setDx(core::MultiVecDerivId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::vec_id::write_access::dx);
    mparams.setDx(v);
}

void MechanicalOperations::setDx(core::ConstMultiVecDerivId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::vec_id::write_access::dx);
    mparams.setDx(v);
}

void MechanicalOperations::setDf(core::MultiVecDerivId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::vec_id::write_access::dforce);
    mparams.setDf(v);
}

void MechanicalOperations::setDf(core::ConstMultiVecDerivId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::vec_id::write_access::dforce);
    mparams.setDf(v);
}


/// Propagate the given displacement through all mappings
void MechanicalOperations::propagateDx(core::MultiVecDerivId dx, bool ignore_flag)
{
    setDx(dx);
    executeVisitor( MechanicalPropagateDxVisitor(&mparams, dx, ignore_flag) );
}

/// Propagate the given displacement through all mappings and reset the current force delta
void MechanicalOperations::propagateDxAndResetDf(core::MultiVecDerivId dx, core::MultiVecDerivId df)
{
    setDx(dx);
    setDf(df);
    executeVisitor( MechanicalPropagateDxAndResetForceVisitor(&mparams, dx, df) );
}

/// Propagate the given position through all mappings
void MechanicalOperations::propagateX(core::MultiVecCoordId x)
{
    setX(x);
    const MechanicalPropagateOnlyPositionVisitor visitor(&mparams, 0.0, x);
    executeVisitor( visitor );
}

/// Propagate the given velocity through all mappings
void MechanicalOperations::propagateV(core::MultiVecDerivId v)
{
    setV(v);
    const MechanicalPropagateOnlyVelocityVisitor visitor(&mparams, 0.0, v);
    executeVisitor( visitor );
}

/// Propagate the given position and velocity through all mappings
void MechanicalOperations::propagateXAndV(core::MultiVecCoordId x, core::MultiVecDerivId v)
{
    setX(x);
    setV(v);
    const MechanicalPropagateOnlyPositionAndVelocityVisitor visitor(&mparams, 0.0, x, v);
    executeVisitor( visitor );
}

/// Propagate the given position through all mappings and reset the current force delta
void MechanicalOperations::propagateXAndResetF(core::MultiVecCoordId x, core::MultiVecDerivId f)
{
    setX(x);
    setF(f);
    const MechanicalPropagateOnlyPositionAndResetForceVisitor visitor(&mparams, x, f);
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
    kineticEnergy = 0;
    potentialEnergy = 0;
    MechanicalComputeEnergyVisitor energyVisitor(&mparams);
    executeVisitor(&energyVisitor);
    kineticEnergy = energyVisitor.getKineticEnergy();
    potentialEnergy = energyVisitor.getPotentialEnergy();
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
void MechanicalOperations::computeForce(core::MultiVecDerivId result, bool clear, bool accumulate)
{
    setF(result);
    if (clear)
    {
        executeVisitor( MechanicalResetForceVisitor(&mparams, result, false) );
        //finish();
    }
    executeVisitor( MechanicalComputeForceVisitor(&mparams, result, accumulate) );
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
    const core::ConstMultiVecDerivId dx = mparams.dx();
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
void MechanicalOperations::addMBKdx(core::MultiVecDerivId df,
                                    const MatricesFactors::M m,
                                    const MatricesFactors::B b,
                                    const MatricesFactors::K k,
                                    const bool clear, const bool accumulate)
{
    setDf(df);
    if (clear)
    {
        executeVisitor( MechanicalResetForceVisitor(&mparams, df, true) );
        //finish();
    }
    mparams.setBFactor(b.get());
    mparams.setKFactor(k.get());
    mparams.setMFactor(m.get());
    executeVisitor( MechanicalAddMBKdxVisitor(&mparams, df, accumulate) );
}

/// accumulate $ df += (m M + b B + k K) velocity $
void MechanicalOperations::addMBKv(core::MultiVecDerivId df,
                                   const MatricesFactors::M m,
                                   const MatricesFactors::B b,
                                   const MatricesFactors::K k,
                                   const bool clear, const bool accumulate)
{
    const core::ConstMultiVecDerivId dx = mparams.dx();
    mparams.setDx(mparams.v());
    setDf(df);
    if (clear)
    {
        executeVisitor( MechanicalResetForceVisitor(&mparams, df, true) );
        //finish();
    }
    mparams.setBFactor(b.get());
    mparams.setKFactor(k.get());
    mparams.setMFactor(m.get());
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
    MultiVecDerivId f( vec_id::write_access::force );
    setF(f);
    setDx(a);
    setX(x);
    setV(v);
    executeVisitor( MechanicalProjectPositionAndVelocityVisitor(&mparams, t,x,v) );
    executeVisitor( MechanicalPropagateOnlyPositionAndVelocityVisitor(&mparams, t,x,v) );
    computeForce(f);

    accFromF(a,f);
    projectResponse(a);
}

void MechanicalOperations::computeForce(SReal t, core::MultiVecDerivId f, core::MultiVecCoordId x, core::MultiVecDerivId v)
{
    setF(f);
    setX(x);
    setV(v);
    executeVisitor( MechanicalProjectPositionAndVelocityVisitor(&mparams, t,x,v) );
    executeVisitor( MechanicalPropagateOnlyPositionAndVelocityVisitor(&mparams, t,x,v) );
    computeForce(f,true,true);

    projectResponse(f);
}

void MechanicalOperations::computeContactAcc(SReal t, core::MultiVecDerivId a, core::MultiVecCoordId x, core::MultiVecDerivId v)
{
    MultiVecDerivId f( vec_id::write_access::force );
    setF(f);
    setDx(a);
    setX(x);
    setV(v);
    executeVisitor( MechanicalProjectPositionAndVelocityVisitor(&mparams, t,x,v) );
    executeVisitor( MechanicalPropagateOnlyPositionAndVelocityVisitor(&mparams, t,x,v) );
    computeContactForce(f);

    accFromF(a,f);
    projectResponse(a);
}

void MechanicalOperations::resetSystem(core::behavior::LinearSolver* linearSolver)
{
    if (linearSolver)
    {
        if (auto* linearSystem = linearSolver->getLinearSystem())
        {
            linearSystem->clearSystem();
        }
    }
}

void MechanicalOperations::setSystemMBKMatrix(
    MatricesFactors::M m, MatricesFactors::B b, MatricesFactors::K k,
    core::behavior::LinearSolver* linearSolver)
{
    if (linearSolver)
    {
        mparams.setMFactor(m.get());
        mparams.setBFactor(b.get());
        mparams.setKFactor(k.get());
        mparams.setSupportOnlySymmetricMatrix(!linearSolver->supportNonSymmetricSystem());
        if (auto* linearSystem = linearSolver->getLinearSystem())
        {
            linearSystem->buildSystemMatrix(&mparams);
        }
    }
}

void MechanicalOperations::setSystemRHVector(
    core::MultiVecDerivId v, core::behavior::LinearSolver* linearSolver)
{
    if (linearSolver)
    {
        if (auto* linearSystem = linearSolver->getLinearSystem())
        {
            linearSystem->setRHS(v);
        }
    }
}

void MechanicalOperations::setSystemLHVector(
    core::MultiVecDerivId v, core::behavior::LinearSolver* linearSolver)
{
    if (linearSolver)
    {
        if (auto* linearSystem = linearSolver->getLinearSystem())
        {
            linearSystem->setSystemSolution(v);
        }
    }
}

void MechanicalOperations::solveSystem(core::behavior::LinearSolver* linearSolver)
{
    if (linearSolver)
    {
        linearSolver->solveSystem();
    }
}

void MechanicalOperations::solveSystem(
    core::behavior::LinearSolver* linearSolver, core::MultiVecDerivId v)
{
    if (linearSolver)
    {
        linearSolver->solveSystem();
        if (auto* linearSystem = linearSolver->getLinearSystem())
        {
            linearSystem->dispatchSystemSolution(v);
        }
    }
}

void MechanicalOperations::print(std::ostream& out,
    core::behavior::LinearSolver* linearSolver)
{
    if (linearSolver)
    {
        const linearalgebra::BaseMatrix* m = linearSolver->getLinearSystem()->getSystemBaseMatrix();
        if (!m)
        {
            return;
        }
        const auto ny = m->rowSize();
        const auto nx = m->colSize();
        out << "[";
        for (linearalgebra::BaseMatrix::Index y = 0; y < ny; ++y)
        {
            out << "[";
            for (linearalgebra::BaseMatrix::Index x = 0; x < nx; x++)
            {
                out << ' ' << m->element(x, y);
            }
            out << "]";
        }
        out << "]";
    }
}


using sofa::core::behavior::LinearSolver;
using sofa::core::objectmodel::BaseContext;

/*
void MechanicalOperations::solveConstraint(SReal dt, MultiVecDerivId id, core::ConstraintParams::ConstOrder order )
{
  core::ConstraintParams cparams(mparams    // PARAMS FIRST //, order);
  mparams.setDt(dt);
  assert( order == core::ConstraintParams::ConstOrder::VEL || order == core::ConstraintParams::ConstOrder::ACC);
  cparams.setV( id);
  solveConstraint(&cparams    // PARAMS FIRST //, id);
}

void MechanicalOperations::solveConstraint(SReal dt, MultiVecCoordId id, core::ConstraintParams::ConstOrder order)
{
  core::ConstraintParams cparams(mparams    // PARAMS FIRST //, order);
  mparams.setDt(dt);
  assert( order == core::ConstraintParams::ConstOrder::POS);
  cparams.setX( id);
  solveConstraint(&cparams    // PARAMS FIRST //, id);
}
*/

void MechanicalOperations::solveConstraint(MultiVecId id, core::ConstraintOrder order)
{
    cparams.setOrder(order);

    type::vector< core::behavior::ConstraintSolver* > constraintSolverList;

    ctx->get<core::behavior::ConstraintSolver>(&constraintSolverList, ctx->getTags(), BaseContext::Local);

    for (auto* constraintSolver : constraintSolverList)
    {
        constraintSolver->solveConstraint(&cparams, id);
    }
}

// BaseMatrix & BaseVector Computations
void MechanicalOperations::getMatrixDimension(sofa::Size*  const nbRow, sofa::Size* const nbCol, sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    executeVisitor( MechanicalGetMatrixDimensionVisitor(&mparams, nbRow, nbCol, matrix) );
}

void MechanicalOperations::addMBK_ToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, SReal mFact, SReal bFact, SReal kFact)
{
    mparams.setMFactor(mFact);
    mparams.setBFactor(bFact);
    mparams.setKFactor(kFact);
    if (matrix != nullptr)
    {
        executeVisitor( MechanicalAddMBK_ToMatrixVisitor(&mparams, matrix) );
        executeVisitor( MechanicalApplyProjectiveConstraint_ToMatrixVisitor(&mparams, matrix) );
    }
}

void MechanicalOperations::baseVector2MultiVector(const linearalgebra::BaseVector *src, core::MultiVecId dest, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    if (src != nullptr)
    {
        executeVisitor( MechanicalMultiVectorFromBaseVectorVisitor(&mparams, dest, src, matrix) );
    }
}

void MechanicalOperations::multiVector2BaseVector(core::ConstMultiVecId src, linearalgebra::BaseVector *dest, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    if (dest != nullptr)
    {
        executeVisitor( MechanicalMultiVectorToBaseVectorVisitor(&mparams, src, dest, matrix) );
    }
}


void MechanicalOperations::multiVectorPeqBaseVector(core::MultiVecDerivId dest, const linearalgebra::BaseVector *src, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    if (src != nullptr)
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

void MechanicalOperations::showMissingLinearSolverError() const
{
    if (!hasShownMissingLinearSolverMap[ctx])
    {
        const auto solvers = sofa::core::ObjectFactory::getInstance()->listClassesDerivedFrom<sofa::core::behavior::BaseLinearSolver>();
        msg_error(ctx) << "A linear solver is required, but has not been found. Add a linear solver to your scene to "
                          "fix this issue. The list of available linear solver components "
                          "is: [" << solvers << "].";
        hasShownMissingLinearSolverMap[ctx] = true;
    }
}

}
