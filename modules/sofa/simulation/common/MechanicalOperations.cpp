#include <sofa/simulation/common/MechanicalOperations.h>

namespace sofa
{

namespace simulation
{

namespace common
{

MechanicalOperations::MechanicalOperations(sofa::core::objectmodel::BaseContext* ctx, const sofa::core::MechanicalParams* mparams)
    :mparams(*mparams),ctx(ctx)
{
}

MechanicalOperations::MechanicalOperations(sofa::core::objectmodel::BaseContext* ctx, const sofa::core::ExecParams* params)
    :mparams(*params),ctx(ctx)
{
}

/// Propagate the given displacement through all mappings
void MechanicalOperations::propagateDx(core::MultiVecDerivId dx)
{
}

/// Propagate the given displacement through all mappings and reset the current force delta
void MechanicalOperations::propagateDxAndResetDf(core::MultiVecDerivId dx, core::MultiVecDerivId df)
{
}

/// Propagate the given position through all mappings
void MechanicalOperations::propagateX(core::MultiVecCoordId x)
{
}

/// Propagate the given position through all mappings and reset the current force delta
void MechanicalOperations::propagateXAndResetF(core::MultiVecCoordId x, core::MultiVecDerivId f)
{
}

/// Apply projective constraints to the given vector
void MechanicalOperations::projectResponse(core::MultiVecDerivId dx, double **W)
{
}

void MechanicalOperations::addMdx(core::MultiVecDerivId res, core::MultiVecDerivId dx(core::VecDerivId() ) , double factor)
{
}

///< res += factor M.dx
void MechanicalOperations::integrateVelocity(core::MultiVecDerivId res, core::ConstMultiVecCoordId x, core::ConstMultiVecDerivId v, double dt)
{
}

///< res = x + v.dt
void MechanicalOperations::accFromF(core::MultiVecDerivId a, core::MultiVecDerivId f) ///< a = M^-1 . f
{
}

/// Compute the current force (given the latest propagated position and velocity)
void MechanicalOperations::computeForce(core::MultiVecDerivId result, bool clear, bool accumulate)
{
}

/// Compute the current force delta (given the latest propagated displacement)
void MechanicalOperations::computeDf(core::MultiVecDerivId df, bool clear, bool accumulate)
{
}

/// Compute the current force delta (given the latest propagated velocity)
void MechanicalOperations::computeDfV(core::MultiVecDerivId df, bool clear, bool accumulate)
{
}

/// accumulate $ df += (m M + b B + k K) dx $ (given the latest propagated displacement)
void MechanicalOperations::addMBKdx(core::MultiVecDerivId df, double m, double b, double k, bool clear, bool accumulate)
{
}

/// accumulate $ df += (m M + b B + k K) velocity $
void MechanicalOperations::addMBKv(core::MultiVecDerivId df, double m, double b, double k, bool clear, bool accumulate)
{
}

/// Add dt*Gravity to the velocity
void MechanicalOperations::addSeparateGravity(double dt, core::MultiVecDerivId result)
{
}

void MechanicalOperations::computeContactForce(core::MultiVecDerivId result)
{
}

void MechanicalOperations::computeContactDf(core::MultiVecDerivId df)
{
}

void MechanicalOperations::computeAcc(double t, core::MultiVecDerivId a, core::MultiVecCoordId x, core::MultiVecDerivId v)
{
}

void MechanicalOperations::computeContactAcc(double t, core::MultiVecDerivId a, core::MultiVecCoordId x, core::MultiVecDerivId v)
{
}

/// @}

/// @name Matrix operations using LinearSolver components
/// @{

void MechanicalOperations::m_resetSystem()
{
}

void MechanicalOperations::m_setSystemMBKMatrix(double mFact, double bFact, double kFact)
{
}

void MechanicalOperations::m_setSystemRHVector(core::MultiVecDerivId v)
{
}

void MechanicalOperations::m_setSystemLHVector(core::MultiVecDerivId v)
{
}

void MechanicalOperations::m_solveSystem()
{
}

void MechanicalOperations::m_print( std::ostream& out )
{
}

/// @}

/// @name Matrix operations
/// @{

// BaseMatrix & BaseVector Computations
void MechanicalOperations::getMatrixDimension(unsigned int * const, unsigned int * const, sofa::core::behavior::MultiMatrixAccessor* matrix)
{
}

void MechanicalOperations::addMBK_ToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, double mFact, double bFact, double kFact)
{
}


/*
void MechanicalOperations::multiVector2BaseVector(core::ConstMultiVecId src, defaulttype::BaseVector *dest, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
}


void MechanicalOperations::multiVectorPeqBaseVector(core::MultiVecId dest, defaulttype::BaseVector *src, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
}
*/


/// @}

/// @name Debug operations
/// @{

/// Dump the content of the given vector.
void MechanicalOperations::print( core::ConstMultiVecId v, std::ostream& out )
{
}


void MechanicalOperations::printWithElapsedTime( core::ConstMultiVecId v,  unsigned time, std::ostream& out )
{
}

/// @}

}

}

}
