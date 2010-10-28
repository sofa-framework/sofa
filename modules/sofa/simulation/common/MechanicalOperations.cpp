#include <sofa/simulation/common/MechanicalOperations.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/MechanicalMatrixVisitor.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/VecId.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/defaulttype/BaseMatrix.h>
namespace sofa
{

namespace simulation
{

namespace common
{

MechanicalOperations::MechanicalOperations(sofa::core::objectmodel::BaseContext* ctx, const sofa::core::MechanicalParams* mparams)
    :mparams(*mparams),ctx(ctx),executeVisitor(*ctx)
{
}

MechanicalOperations::MechanicalOperations(sofa::core::objectmodel::BaseContext* ctx, const sofa::core::ExecParams* params)
    :mparams(*params),ctx(ctx),executeVisitor(*ctx)
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
    if (v.getDefaultId().isNull()) v.setDefaultId(core::VecDerivId::force());
    mparams.setDf(v);
}

void MechanicalOperations::setDf(core::ConstMultiVecDerivId& v)
{
    if (v.getDefaultId().isNull()) v.setDefaultId(core::VecDerivId::force());
    mparams.setDf(v);
}


/// Propagate the given displacement through all mappings
void MechanicalOperations::propagateDx(core::MultiVecDerivId dx)
{
    setDx(dx);
    executeVisitor( MechanicalPropagateDxVisitor(dx, false, false, &mparams) );
}

/// Propagate the given displacement through all mappings and reset the current force delta
void MechanicalOperations::propagateDxAndResetDf(core::MultiVecDerivId dx, core::MultiVecDerivId df)
{
    setDx(dx);
    setDf(df);
    executeVisitor( MechanicalPropagateDxAndResetForceVisitor(dx, df, false, &mparams) );
}

/// Propagate the given position through all mappings
void MechanicalOperations::propagateX(core::MultiVecCoordId x)
{
    setX(x);
    executeVisitor( MechanicalPropagateXVisitor(x, false, &mparams) //Don't ignore the masks
                  );
}

/// Propagate the given position through all mappings and reset the current force delta
void MechanicalOperations::propagateXAndResetF(core::MultiVecCoordId x, core::MultiVecDerivId f)
{
    setX(x);
    setF(f);
    executeVisitor( MechanicalPropagateXAndResetForceVisitor(x, f, false, &mparams) );
}

/// Apply projective constraints to the given vector
void MechanicalOperations::projectResponse(core::MultiVecDerivId dx, double **W)
{
    setDx(dx);
    executeVisitor( MechanicalApplyConstraintsVisitor(dx, W, &mparams) );
}

void MechanicalOperations::addMdx(core::MultiVecDerivId res, core::MultiVecDerivId dx, double factor)
{
    setDx(dx);
    executeVisitor( MechanicalAddMDxVisitor(res,dx,factor,&mparams) );
}

///< res += factor M.dx
void MechanicalOperations::integrateVelocity(core::MultiVecDerivId res, core::ConstMultiVecCoordId x, core::ConstMultiVecDerivId v, double dt)
{
    executeVisitor( MechanicalVOpVisitor(res,x,v,dt, &mparams) );
}

///< res = x + v.dt
void MechanicalOperations::accFromF(core::MultiVecDerivId a, core::ConstMultiVecDerivId f) ///< a = M^-1 . f
{
    setDx(a);
    setF(f);

    executeVisitor( MechanicalAccFromFVisitor(a,&mparams) );
}

/// Compute the current force (given the latest propagated position and velocity)
void MechanicalOperations::computeForce(core::MultiVecDerivId result, bool clear, bool accumulate)
{
    setF(result);
    if (clear)
    {
        executeVisitor( MechanicalResetForceVisitor(result, false, &mparams), true ); // enable prefetching
        //finish();
    }
    executeVisitor( MechanicalComputeForceVisitor(result, accumulate, &mparams) , true ); // enable prefetching
}

/// Compute the current force delta (given the latest propagated displacement)
void MechanicalOperations::computeDf(core::MultiVecDerivId df, bool clear, bool accumulate)
{
    setDf(df);
    if (clear)
    {
        executeVisitor( MechanicalResetForceVisitor(df, false, &mparams) );
        //	finish();
    }
    executeVisitor( MechanicalComputeDfVisitor( df,  accumulate, &mparams) );
}

/// Compute the current force delta (given the latest propagated velocity)
void MechanicalOperations::computeDfV(core::MultiVecDerivId df, bool clear, bool accumulate)
{
    core::ConstMultiVecDerivId dx = mparams.dx();
    mparams.setDx(mparams.v());
    setDf(df);
    if (clear)
    {
        executeVisitor( MechanicalResetForceVisitor(df, false, &mparams) );
        //finish();
    }
    executeVisitor( MechanicalComputeDfVisitor(df, accumulate, &mparams) );
    mparams.setDx(dx);
}

/// accumulate $ df += (m M + b B + k K) dx $ (given the latest propagated displacement)
void MechanicalOperations::addMBKdx(core::MultiVecDerivId df, double m, double b, double k, bool clear, bool accumulate)
{
    setDf(df);
    if (clear)
    {
        executeVisitor( MechanicalResetForceVisitor(df, true, &mparams) );
        //finish();
    }
    mparams.setBFactor(b);
    mparams.setKFactor(k);
    mparams.setMFactor(m);
    executeVisitor( MechanicalAddMBKdxVisitor(df, accumulate, &mparams) );
}

/// accumulate $ df += (m M + b B + k K) velocity $
void MechanicalOperations::addMBKv(core::MultiVecDerivId df, double m, double b, double k, bool clear, bool accumulate)
{
    core::ConstMultiVecDerivId dx = mparams.dx();
    mparams.setDx(mparams.v());
    setDf(df);
    if (clear)
    {
        executeVisitor( MechanicalResetForceVisitor(df, true, &mparams) );
        //finish();
    }
    mparams.setBFactor(b);
    mparams.setKFactor(k);
    mparams.setMFactor(m);
    /* useV = true */
    executeVisitor( MechanicalAddMBKdxVisitor(df, accumulate, &mparams) );
    mparams.setDx(dx);
}

/// Add dt*Gravity to the velocity
void MechanicalOperations::addSeparateGravity(double dt, core::MultiVecDerivId result)
{
    mparams.setDt(dt);
    setV(result);
    executeVisitor( MechanicalAddSeparateGravityVisitor(result, &mparams) );
}

void MechanicalOperations::computeContactForce(core::MultiVecDerivId result)
{
    setF(result);
    executeVisitor( MechanicalResetForceVisitor(result, false, &mparams) );
    //finish();
    executeVisitor( MechanicalComputeContactForceVisitor(result, &mparams) );
}

void MechanicalOperations::computeContactDf(core::MultiVecDerivId df)
{
    setDf(df);
    executeVisitor( MechanicalResetForceVisitor(df, false, &mparams) );
    //finish();
}

void MechanicalOperations::computeAcc(double t, core::MultiVecDerivId a, core::MultiVecCoordId x, core::MultiVecDerivId v)
{
    MultiVecDerivId f( VecDerivId::force() );
    setF(f);
    setDx(a);
    setX(x);
    setV(v);
    executeVisitor( MechanicalPropagatePositionAndVelocityVisitor(t,x,v,
#ifdef SOFA_SUPPORT_MAPPED_MASS
            a,
#endif
            true,&mparams) );
    computeForce(f);

    accFromF(a,f);
    projectResponse(a);
}

void MechanicalOperations::computeContactAcc(double t, core::MultiVecDerivId a, core::MultiVecCoordId x, core::MultiVecDerivId v)
{
    MultiVecDerivId f( VecDerivId::force() );
    setF(f);
    setDx(a);
    setX(x);
    setV(v);
    executeVisitor( MechanicalPropagatePositionAndVelocityVisitor(t,x,v,
#ifdef SOFA_SUPPORT_MAPPED_MASS
            a,
#endif
            true,&mparams) );
    computeContactForce(f);

    accFromF(a,f);
    projectResponse(a);
}

/// @}

/// @name Matrix operations using LinearSolver components
/// @{

using sofa::core::behavior::LinearSolver;
using sofa::core::objectmodel::BaseContext;

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

void MechanicalOperations::m_setSystemMBKMatrix(double mFact, double bFact, double kFact)
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

/// @}

/// @name Matrix operations
/// @{

// BaseMatrix & BaseVector Computations
void MechanicalOperations::getMatrixDimension(unsigned int *  const nbRow, unsigned int * const nbCol, sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    executeVisitor( MechanicalGetMatrixDimensionVisitor(nbRow, nbCol, matrix,&mparams) );
}

void MechanicalOperations::addMBK_ToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, double mFact, double bFact, double kFact)
{
    mparams.setMFactor(mFact);
    mparams.setBFactor(bFact);
    mparams.setKFactor(kFact);
    if (matrix != NULL)
    {
        //std::cout << "MechanicalAddMBK_ToMatrixVisitor "<< mFact << " " << bFact << " " << kFact << " " << offset << std::endl;
        executeVisitor( MechanicalAddMBK_ToMatrixVisitor(matrix,&mparams) );
    }
}



void MechanicalOperations::multiVector2BaseVector(core::ConstMultiVecId src, defaulttype::BaseVector *dest, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    if (dest != NULL)
    {
        executeVisitor( MechanicalMultiVectorToBaseVectorVisitor(src, dest, matrix, &mparams) );
    }
}


void MechanicalOperations::multiVectorPeqBaseVector(core::MultiVecDerivId dest, defaulttype::BaseVector *src, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    if (src != NULL)
    {
        executeVisitor( MechanicalMultiVectorPeqBaseVectorVisitor(dest, src, matrix, &mparams) );
    }
}



/// @}

/// @name Debug operations
/// @{

/// Dump the content of the given vector.
void MechanicalOperations::print( core::ConstMultiVecId /*v*/, std::ostream& /*out*/ )
{
}


void MechanicalOperations::printWithElapsedTime( core::ConstMultiVecId /*v*/, unsigned /*time*/, std::ostream& /*out*/ )
{
}

/// @}

}

}

}
