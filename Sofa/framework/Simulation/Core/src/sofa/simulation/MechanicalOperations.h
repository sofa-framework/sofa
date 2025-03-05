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
#ifndef SOFA_SIMULATION_CORE_MECHANICALOPERATIONS_H
#define SOFA_SIMULATION_CORE_MECHANICALOPERATIONS_H

#include <sofa/simulation/config.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/simulation/VisitorExecuteFunc.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MatricesFactors.h>


namespace sofa::simulation::common
{

class SOFA_SIMULATION_CORE_API MechanicalOperations
{
public:
    core::MechanicalParams mparams;
    core::ConstraintParams cparams;
    core::objectmodel::BaseContext* ctx;

    MechanicalOperations(const core::MechanicalParams* mparams, core::objectmodel::BaseContext* ctx, bool precomputedTraversalOrder=false);

    MechanicalOperations(const core::ExecParams* params, core::objectmodel::BaseContext* ctx, bool precomputedTraversalOrder=false);

    core::MechanicalParams* operator->() { return &mparams; }
    operator const core::MechanicalParams*() { return &mparams; }

    /// @name Mechanical Vector operations
/// @{

    /// Propagate the given displacement through all mappings
    void propagateDx(core::MultiVecDerivId dx, bool ignore_flag = false);
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
    /// Apply projective constraints to the given position vector
    void projectPosition(core::MultiVecCoordId x, SReal time = 0.0);
    /// Apply projective constraints to the given velocity vector
    void projectVelocity(core::MultiVecDerivId v, SReal time = 0.0);
    /// Apply projective constraints to the given vector
    void projectResponse(core::MultiVecDerivId dx, double **W=nullptr);
    /// Apply projective constraints to the given position and velocity vectors
    void projectPositionAndVelocity(core::MultiVecCoordId x, core::MultiVecDerivId v, double time = 0.0);
    void addMdx(core::MultiVecDerivId res, core::MultiVecDerivId dx, SReal factor = 1.0); ///< res += factor M.dx
    void integrateVelocity(core::MultiVecDerivId res, core::ConstMultiVecCoordId x, core::ConstMultiVecDerivId v, SReal dt); ///< res = x + v.dt
    void accFromF(core::MultiVecDerivId a, core::ConstMultiVecDerivId f); ///< a = M^-1 . f
    /// Compute Energy
    void computeEnergy(SReal &kineticEnergy, SReal &potentialEnergy);
    /// Compute the current force (given the latest propagated position and velocity)
    void computeForce(core::MultiVecDerivId result, bool clear = true, bool accumulate = true);
    /// Compute the current force delta (given the latest propagated displacement)
    void computeDf(core::MultiVecDerivId df, bool clear = true, bool accumulate = true);
    /// Compute the current force delta (given the latest propagated velocity)
    void computeDfV(core::MultiVecDerivId df, bool clear = true, bool accumulate = true);
    /// accumulate $ df += (m M + b B + k K) dx $ (given the latest propagated displacement)
    void addMBKdx(core::MultiVecDerivId df, core::MatricesFactors::M m, core::MatricesFactors::B b, core::MatricesFactors::K k, bool clear = true, bool accumulate = true);
    /// accumulate $ df += (m M + b B + k K) velocity $
    void addMBKv(core::MultiVecDerivId df, core::MatricesFactors::M m, core::MatricesFactors::B b, core::MatricesFactors::K k, bool clear = true, bool accumulate = true);
    /// Add dt*Gravity to the velocity
    void addSeparateGravity(SReal dt, core::MultiVecDerivId result = core::vec_id::write_access::velocity );

    void computeContactForce(core::MultiVecDerivId result);
    void computeContactDf(core::MultiVecDerivId df);


    void computeAcc(SReal t, core::MultiVecDerivId a, core::MultiVecCoordId x, core::MultiVecDerivId v); ///< Compute a(x,v) at time t. Parameters x and v not const due to propagation through mappings.
    void computeForce(SReal t, core::MultiVecDerivId f, core::MultiVecCoordId x, core::MultiVecDerivId v);  ///< Compute f(x,v) at time t. Parameters x and v not const due to propagation through mappings.
    void computeContactAcc(SReal t, core::MultiVecDerivId a, core::MultiVecCoordId x, core::MultiVecDerivId v); // Parameters x and v not const due to propagation through mappings.

    /// @}

    /// @name Matrix operations using LinearSolver components
/// @{

    void resetSystem(core::behavior::LinearSolver* linearSolver);
    void setSystemMBKMatrix(core::MatricesFactors::M m, core::MatricesFactors::B b, core::MatricesFactors::K k, core::behavior::LinearSolver* linearSolver);
    void setSystemRHVector(core::MultiVecDerivId v, core::behavior::LinearSolver* linearSolver);
    void setSystemLHVector(core::MultiVecDerivId v, core::behavior::LinearSolver* linearSolver);
    void solveSystem(core::behavior::LinearSolver* linearSolver);
    void print( std::ostream& out, core::behavior::LinearSolver* linearSolver);

    SOFA_ATTRIBUTE_DEPRECATED_MECHANICALOPERATIONS_RESETSYSTEM() void m_resetSystem();
    SOFA_ATTRIBUTE_DEPRECATED_MECHANICALOPERATIONS_SETSYSTEMMBKMATRIX() void m_setSystemMBKMatrix(SReal mFact, SReal bFact, SReal kFact);
    SOFA_ATTRIBUTE_DEPRECATED_MECHANICALOPERATIONS_SETSYSTEMRHVECTOR() void m_setSystemRHVector(core::MultiVecDerivId v);
    SOFA_ATTRIBUTE_DEPRECATED_MECHANICALOPERATIONS_SETSYSTEMLHVECTOR() void m_setSystemLHVector(core::MultiVecDerivId v);
    SOFA_ATTRIBUTE_DEPRECATED_MECHANICALOPERATIONS_SOLVESYSTEM() void m_solveSystem();
    SOFA_ATTRIBUTE_DEPRECATED_MECHANICALOPERATIONS_PRINT() void m_print( std::ostream& out );

    /// @}

    /** Find all the Constraint present in the scene graph, build the constraint equation system, solve and apply the correction
**/
    void solveConstraint(sofa::core::MultiVecId id, core::ConstraintOrder order);



    /// @name Matrix operations
/// @{

    // BaseMatrix & BaseVector Computations
    void getMatrixDimension(sofa::Size* const, sofa::Size* const, sofa::core::behavior::MultiMatrixAccessor* matrix = nullptr);
    void getMatrixDimension(sofa::core::behavior::MultiMatrixAccessor* matrix)
    {
        getMatrixDimension(nullptr, nullptr, matrix);
    }

    void addMBK_ToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, SReal mFact, SReal bFact, SReal kFact);

    void multiVector2BaseVector(core::ConstMultiVecId src, linearalgebra::BaseVector *dest, const sofa::core::behavior::MultiMatrixAccessor* matrix);
    void baseVector2MultiVector(const linearalgebra::BaseVector *src, core::MultiVecId dest, const sofa::core::behavior::MultiMatrixAccessor* matrix);
    void multiVectorPeqBaseVector(core::MultiVecDerivId dest, const linearalgebra::BaseVector *src, const sofa::core::behavior::MultiMatrixAccessor* matrix);

    /// @}

    /// @name Debug operations
/// @{

    /// Dump the content of the given vector.
    void print( core::ConstMultiVecId v, std::ostream& out );
    void printWithElapsedTime( core::ConstMultiVecId v,  unsigned time, std::ostream& out=std::cerr );

    /// @}


    SOFA_ATTRIBUTE_DEPRECATED_MECHANICALOPERATIONS_ADDMBKDX()
    void addMBKdx(core::MultiVecDerivId df, SReal m, SReal b, SReal k, bool clear = true, bool accumulate = true);
    SOFA_ATTRIBUTE_DEPRECATED_MECHANICALOPERATIONS_ADDMBKV()
    void addMBKv(core::MultiVecDerivId df, SReal m, SReal b, SReal k, bool clear = true, bool accumulate = true);
    SOFA_ATTRIBUTE_DEPRECATED_MECHANICALOPERATIONS_SETSYSTEMMBKMATRIX_OTHER()
    void setSystemMBKMatrix(SReal mFact, SReal bFact, SReal kFact, core::behavior::LinearSolver* linearSolver);

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

    /// Warn the user that a linear solver is required but has not been found
    void showMissingLinearSolverError() const;

    /// Store if the "missing linear solver" error message has already been shown for a given context
    static std::map<core::objectmodel::BaseContext*, bool> hasShownMissingLinearSolverMap;

};

}

#endif //SOFA_SIMULATION_CORE_MECHANICALOPERATIONS_H
