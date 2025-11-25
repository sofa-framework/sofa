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
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/MatricesFactors.h>


namespace sofa::simulation::common
{

class SOFA_SIMULATION_CORE_API MechanicalOperations
{
public:
    sofa::core::MechanicalParams mparams;
    sofa::core::ConstraintParams cparams;
    sofa::core::objectmodel::BaseContext* ctx;

    MechanicalOperations(const sofa::core::MechanicalParams* mparams, sofa::core::objectmodel::BaseContext* ctx, bool precomputedTraversalOrder=false);

    MechanicalOperations(const sofa::core::ExecParams* params, sofa::core::objectmodel::BaseContext* ctx, bool precomputedTraversalOrder=false);

    sofa::core::MechanicalParams* operator->() { return &mparams; }
    operator const sofa::core::MechanicalParams*() { return &mparams; }

    /// @name Mechanical Vector operations
/// @{

    void apply(sofa::core::MultiVecCoordId out, sofa::core::ConstMultiVecCoordId in, bool filterNonMechanical = true);
    void applyJ(sofa::core::MultiVecDerivId out, sofa::core::ConstMultiVecDerivId in, bool filterNonMechanical = true);
    void applyJT(sofa::core::MultiVecDerivId in, sofa::core::ConstMultiVecDerivId out, bool filterNonMechanical = true);

    /// Propagate the given displacement through all mappings
    void propagateDx(sofa::core::MultiVecDerivId dx, bool ignore_flag = false, bool filterNonMechanical = true);
    /// Propagate the given displacement through all mappings and reset the current force delta
    void propagateDxAndResetDf(sofa::core::MultiVecDerivId dx, sofa::core::MultiVecDerivId df, bool filterNonMechanical = true);
    /// Propagate the given position through all mappings
    void propagateX(sofa::core::MultiVecCoordId x, bool filterNonMechanical = true);
    /// Propagate the given velocity through all mappings
    void propagateV(sofa::core::MultiVecDerivId v, bool filterNonMechanical = true);
    /// Propagate the given position and velocity through all mappings
    void propagateXAndV(sofa::core::MultiVecCoordId x, sofa::core::MultiVecDerivId v, bool filterNonMechanical = true);
    /// Propagate the given position through all mappings and reset the current force delta
    void propagateXAndResetF(sofa::core::MultiVecCoordId x, sofa::core::MultiVecDerivId f);
    /// Apply projective constraints to the given position vector
    void projectPosition(sofa::core::MultiVecCoordId x, SReal time = 0.0);
    /// Apply projective constraints to the given velocity vector
    void projectVelocity(sofa::core::MultiVecDerivId v, SReal time = 0.0);
    /// Apply projective constraints to the given vector
    void projectResponse(sofa::core::MultiVecDerivId dx, double **W=nullptr);
    /// Apply projective constraints to the given position and velocity vectors
    void projectPositionAndVelocity(sofa::core::MultiVecCoordId x, sofa::core::MultiVecDerivId v, double time = 0.0);
    void addMdx(sofa::core::MultiVecDerivId res, sofa::core::MultiVecDerivId dx, SReal factor = 1.0); ///< res += factor M.dx
    void integrateVelocity(sofa::core::MultiVecDerivId res, sofa::core::ConstMultiVecCoordId x, sofa::core::ConstMultiVecDerivId v, SReal dt); ///< res = x + v.dt
    void accFromF(sofa::core::MultiVecDerivId a, sofa::core::ConstMultiVecDerivId f); ///< a = M^-1 . f
    /// Compute Energy
    void computeEnergy(SReal &kineticEnergy, SReal &potentialEnergy);
    /// Compute the current force (given the latest propagated position and velocity)
    void computeForce(sofa::core::MultiVecDerivId result, bool clear = true, bool accumulate = true);
    /// Compute the current force delta (given the latest propagated displacement)
    void computeDf(sofa::core::MultiVecDerivId df, bool clear = true, bool accumulate = true);
    /// Compute the current force delta (given the latest propagated velocity)
    void computeDfV(sofa::core::MultiVecDerivId df, bool clear = true, bool accumulate = true);
    /// accumulate $ df += (m M + b B + k K) dx $ (given the latest propagated displacement)
    void addMBKdx(sofa::core::MultiVecDerivId df, sofa::core::MatricesFactors::M m, sofa::core::MatricesFactors::B b, sofa::core::MatricesFactors::K k, bool clear = true, bool accumulate = true);
    /// accumulate $ df += (m M + b B + k K) velocity $
    void addMBKv(sofa::core::MultiVecDerivId df, sofa::core::MatricesFactors::M m, sofa::core::MatricesFactors::B b, sofa::core::MatricesFactors::K k, bool clear = true, bool accumulate = true);
    /// Add dt*Gravity to the velocity
    void addSeparateGravity(SReal dt, sofa::core::MultiVecDerivId result = sofa::core::vec_id::write_access::velocity );

    void computeContactForce(sofa::core::MultiVecDerivId result);
    void computeContactDf(sofa::core::MultiVecDerivId df);


    void computeAcc(SReal t, sofa::core::MultiVecDerivId a, sofa::core::MultiVecCoordId x, sofa::core::MultiVecDerivId v); ///< Compute a(x,v) at time t. Parameters x and v not const due to propagation through mappings.
    void computeForce(SReal t, sofa::core::MultiVecDerivId f, sofa::core::MultiVecCoordId x, sofa::core::MultiVecDerivId v);  ///< Compute f(x,v) at time t. Parameters x and v not const due to propagation through mappings.
    void computeContactAcc(SReal t, sofa::core::MultiVecDerivId a, sofa::core::MultiVecCoordId x, sofa::core::MultiVecDerivId v); // Parameters x and v not const due to propagation through mappings.

    /// @}

    /// @name Matrix operations using LinearSolver components
/// @{

    void resetSystem(sofa::core::behavior::LinearSolver* linearSolver);
    void setSystemMBKMatrix(sofa::core::MatricesFactors::M m, sofa::core::MatricesFactors::B b, sofa::core::MatricesFactors::K k, sofa::core::behavior::LinearSolver* linearSolver);
    void setSystemRHVector(sofa::core::MultiVecDerivId v, sofa::core::behavior::LinearSolver* linearSolver);
    void setSystemLHVector(sofa::core::MultiVecDerivId v, sofa::core::behavior::LinearSolver* linearSolver);
    void solveSystem(sofa::core::behavior::LinearSolver* linearSolver);
    void solveSystem(core::behavior::LinearSolver* linearSolver, core::MultiVecDerivId v);
    void print( std::ostream& out, sofa::core::behavior::LinearSolver* linearSolver);
    /// @}

    /** Find all the Constraint present in the scene graph, build the constraint equation system, solve and apply the correction
**/
    void solveConstraint(sofa::core::MultiVecId id, sofa::core::ConstraintOrder order);



    /// @name Matrix operations
/// @{

    // BaseMatrix & BaseVector Computations
    void getMatrixDimension(sofa::Size* const, sofa::Size* const, sofa::core::behavior::MultiMatrixAccessor* matrix = nullptr);
    void getMatrixDimension(sofa::core::behavior::MultiMatrixAccessor* matrix)
    {
        getMatrixDimension(nullptr, nullptr, matrix);
    }

    void addMBK_ToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, SReal mFact, SReal bFact, SReal kFact);

    void multiVector2BaseVector(sofa::core::ConstMultiVecId src, linearalgebra::BaseVector *dest, const sofa::core::behavior::MultiMatrixAccessor* matrix);
    void baseVector2MultiVector(const linearalgebra::BaseVector *src, sofa::core::MultiVecId dest, const sofa::core::behavior::MultiMatrixAccessor* matrix);
    void multiVectorPeqBaseVector(sofa::core::MultiVecDerivId dest, const linearalgebra::BaseVector *src, const sofa::core::behavior::MultiMatrixAccessor* matrix);

    /// @}

    /// @name Debug operations
/// @{

    /// Dump the content of the given vector.
    void print( sofa::core::ConstMultiVecId v, std::ostream& out );
    void printWithElapsedTime( sofa::core::ConstMultiVecId v,  unsigned time, std::ostream& out=std::cerr );

    /// @}


    SOFA_ATTRIBUTE_DISABLED_MECHANICALOPERATIONS_ADDMBKDX()
    void addMBKdx(sofa::core::MultiVecDerivId df, SReal m, SReal b, SReal k, bool clear = true, bool accumulate = true) = delete;
    SOFA_ATTRIBUTE_DISABLED_MECHANICALOPERATIONS_ADDMBKV()
    void addMBKv(sofa::core::MultiVecDerivId df, SReal m, SReal b, SReal k, bool clear = true, bool accumulate = true) = delete;
    SOFA_ATTRIBUTE_DISABLED_MECHANICALOPERATIONS_SETSYSTEMMBKMATRIX_OTHER()
    void setSystemMBKMatrix(SReal mFact, SReal bFact, SReal kFact, sofa::core::behavior::LinearSolver* linearSolver) = delete;

protected:
    VisitorExecuteFunc executeVisitor;

    void setX(sofa::core::MultiVecCoordId& v);
    void setX(sofa::core::ConstMultiVecCoordId& v);
    void setV(sofa::core::MultiVecDerivId& v);
    void setV(sofa::core::ConstMultiVecDerivId& v);
    void setF(sofa::core::MultiVecDerivId& v);
    void setF(sofa::core::ConstMultiVecDerivId& v);
    void setDx(sofa::core::MultiVecDerivId& v);
    void setDx(sofa::core::ConstMultiVecDerivId& v);
    void setDf(sofa::core::MultiVecDerivId& v);
    void setDf(sofa::core::ConstMultiVecDerivId& v);

    /// Warn the user that a linear solver is required but has not been found
    void showMissingLinearSolverError() const;

    /// Store if the "missing linear solver" error message has already been shown for a given context
    static std::map<sofa::core::objectmodel::BaseContext*, bool> hasShownMissingLinearSolverMap;

};

}

#endif //SOFA_SIMULATION_CORE_MECHANICALOPERATIONS_H
