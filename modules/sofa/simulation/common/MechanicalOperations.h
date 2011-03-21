#ifndef SOFA_SIMULATION_COMMON_MECHANICALOPERATIONS_H
#define SOFA_SIMULATION_COMMON_MECHANICALOPERATIONS_H

#include <sofa/simulation/common/common.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/simulation/common/VisitorExecuteFunc.h>

namespace sofa
{

namespace simulation
{

namespace common
{

class SOFA_SIMULATION_COMMON_API MechanicalOperations
{
public:
    core::MechanicalParams mparams;
    core::ConstraintParams cparams;
    core::objectmodel::BaseContext* ctx;

    MechanicalOperations(const core::MechanicalParams* mparams /* PARAMS FIRST  = core::MechanicalParams::defaultInstance()*/, core::objectmodel::BaseContext* ctx);

    MechanicalOperations(const core::ExecParams* params /* PARAMS FIRST */, core::objectmodel::BaseContext* ctx);

    core::MechanicalParams* operator->() { return &mparams; }
    operator const core::MechanicalParams*() { return &mparams; }

    /// @name Mechanical Vector operations
    /// @{

    /// Propagate the given displacement through all mappings
    void propagateDx(core::MultiVecDerivId dx);
    /// Propagate the given displacement through all mappings and reset the current force delta
    void propagateDxAndResetDf(core::MultiVecDerivId dx, core::MultiVecDerivId df);
    /// Propagate the given position through all mappings
    void propagateX(core::MultiVecCoordId x);
    /// Propagate the given velocity through all mappings
    void propagateV(core::MultiVecDerivId v);
    /// Propagate the given position and velocity through all mappings
    void propagateXAndV(core::MultiVecCoordId x, core::MultiVecDerivId v);
    /// Propagate the given position through all mappings and reset the current force delta
    void propagateXAndResetF(core::MultiVecCoordId x, core::MultiVecDerivId f);
    /// Apply projective constraints to the given vector
    void projectResponse(core::MultiVecDerivId dx, double **W=NULL);
    void addMdx(core::MultiVecDerivId res, core::MultiVecDerivId dx, double factor = 1.0); ///< res += factor M.dx
    void integrateVelocity(core::MultiVecDerivId res, core::ConstMultiVecCoordId x, core::ConstMultiVecDerivId v, double dt); ///< res = x + v.dt
    void accFromF(core::MultiVecDerivId a, core::ConstMultiVecDerivId f); ///< a = M^-1 . f

    /// Compute the current force (given the latest propagated position and velocity)
    void computeForce(core::MultiVecDerivId result, bool clear = true, bool accumulate = true);
    /// Compute the current force delta (given the latest propagated displacement)
    void computeDf(core::MultiVecDerivId df, bool clear = true, bool accumulate = true);
    /// Compute the current force delta (given the latest propagated velocity)
    void computeDfV(core::MultiVecDerivId df, bool clear = true, bool accumulate = true);
    /// accumulate $ df += (m M + b B + k K) dx $ (given the latest propagated displacement)
    void addMBKdx(core::MultiVecDerivId df, double m, double b, double k, bool clear = true, bool accumulate = true);
    /// accumulate $ df += (m M + b B + k K) velocity $
    void addMBKv(core::MultiVecDerivId df, double m, double b, double k, bool clear = true, bool accumulate = true);
    /// Add dt*Gravity to the velocity
    void addSeparateGravity(double dt, core::MultiVecDerivId result = core::VecDerivId::velocity() );

    void computeContactForce(core::MultiVecDerivId result);
    void computeContactDf(core::MultiVecDerivId df);


    void computeAcc(double t, core::MultiVecDerivId a, core::MultiVecCoordId x, core::MultiVecDerivId v);
    void computeContactAcc(double t, core::MultiVecDerivId a, core::MultiVecCoordId x, core::MultiVecDerivId v);

    /// @}

    /// @name Matrix operations using LinearSolver components
    /// @{

    void m_resetSystem();
    void m_setSystemMBKMatrix(double mFact, double bFact, double kFact);
    void m_setSystemRHVector(core::MultiVecDerivId v);
    void m_setSystemLHVector(core::MultiVecDerivId v);
    void m_solveSystem();
    void m_print( std::ostream& out );

    /// @}

    /** Find all the Constraint present in the scene graph, build the constraint equation system, solve and apply the correction
    **/
    void solveConstraint(MultiVecId id, core::ConstraintParams::ConstOrder order);



    /// @name Matrix operations
    /// @{

    // BaseMatrix & BaseVector Computations
    void getMatrixDimension(unsigned int * const, unsigned int * const, sofa::core::behavior::MultiMatrixAccessor* matrix = NULL);
    void getMatrixDimension(sofa::core::behavior::MultiMatrixAccessor* matrix)
    {
        getMatrixDimension(NULL, NULL, matrix);
    }

    void addMBK_ToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, double mFact, double bFact, double kFact);

    void multiVector2BaseVector(core::ConstMultiVecId src, defaulttype::BaseVector *dest, const sofa::core::behavior::MultiMatrixAccessor* matrix);
    void multiVectorPeqBaseVector(core::MultiVecDerivId dest, defaulttype::BaseVector *src, const sofa::core::behavior::MultiMatrixAccessor* matrix);

    /// @}

    /// @name Debug operations
    /// @{

    /// Dump the content of the given vector.
    void print( core::ConstMultiVecId v, std::ostream& out );
    void printWithElapsedTime( core::ConstMultiVecId v,  unsigned time, std::ostream& out=std::cerr );

    /// @}

protected:
    VisitorExecuteFunc executeVisitor;

    void setX(core::MultiVecCoordId& v);
    void setX(core::ConstMultiVecCoordId& v);
    void setV(core::MultiVecDerivId& v);
    void setV(core::ConstMultiVecDerivId& v);
    void setF(core::MultiVecDerivId& v);
    void setF(core::ConstMultiVecDerivId& v);
    void setDx(core::MultiVecDerivId& v);
    void setDx(core::ConstMultiVecDerivId& v);
    void setDf(core::MultiVecDerivId& v);
    void setDf(core::ConstMultiVecDerivId& v);

};

}

}

}

#endif //SOFA_SIMULATION_COMMON_MECHANICALOPERATIONS_H
