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
#pragma once
#include <sofa/component/constraint/lagrangian/solver/config.h>

#include <sofa/component/constraint/lagrangian/solver/ConstraintSolverImpl.h>
#include <sofa/core/behavior/BaseConstraintCorrection.h>
#include <sofa/core/behavior/BaseLagrangianConstraint.h>

#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/fwd.h>

#include <sofa/linearalgebra/FullMatrix.h>
#include <sofa/linearalgebra/SparseMatrix.h>

#include <sofa/helper/map.h>
#include <sofa/helper/LCPcalc.h>

namespace sofa::component::constraint::lagrangian::solver
{

class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API LCPConstraintProblem : public ConstraintProblem
{
public:
    SReal mu;

    void solveTimed(SReal tolerance, int maxIt, SReal timeout) override;
};

class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API LCPConstraintSolver : public ConstraintSolverImpl
{
public:
    SOFA_CLASS(LCPConstraintSolver, ConstraintSolverImpl);

    typedef sofa::core::MultiVecId MultiVecId;

protected:
    /**
    * @brief Default Constructor
    */
    LCPConstraintSolver();

    /**
    * @brief Default Destructor
    */
    ~LCPConstraintSolver() override;
public:

    bool prepareStates(const core::ConstraintParams * /*cParams*/, MultiVecId res1, MultiVecId res2=MultiVecId::null()) override;
    bool buildSystem(const core::ConstraintParams * /*cParams*/, MultiVecId res1, MultiVecId res2=MultiVecId::null()) override;
    bool solveSystem(const core::ConstraintParams * /*cParams*/, MultiVecId res1, MultiVecId res2=MultiVecId::null()) override;
    bool applyCorrection(const core::ConstraintParams * /*cParams*/, MultiVecId res1, MultiVecId res2=MultiVecId::null()) override;



    Data<bool> d_displayDebug; ///< Display debug information.
    Data<bool> d_initial_guess; ///< activate LCP results history to improve its resolution performances.
    Data<bool> d_build_lcp; ///< LCP is not fully built to increase performance in some case.
    Data<SReal> d_tol; ///< residual error threshold for termination of the Gauss-Seidel algorithm
    Data<int> d_maxIt; ///< maximal number of iterations of the Gauss-Seidel algorithm
    Data<SReal> d_regularizationTerm; ///< add regularization*Id to W when solving for constraints
    Data<SReal> d_mu; ///< Friction coefficient
    Data<SReal> d_minW; ///< If not zero, constraints whose self-compliance (i.e. the corresponding value on the diagonal of W) is smaller than this threshold will be ignored
    Data<SReal> d_maxF; ///< If not zero, constraints whose response force becomes larger than this threshold will be ignored
    DeprecatedAndRemoved d_multi_grid; ///< activate multi_grid resolution (NOT STABLE YET)
    DeprecatedAndRemoved d_multi_grid_levels; ///< if multi_grid is active: how many levels to create (>=2)
    DeprecatedAndRemoved d_merge_method; ///< if multi_grid is active: which method to use to merge constraints (0 = compliance-based, 1 = spatial coordinates)
    DeprecatedAndRemoved d_merge_spatial_step; ///< if merge_method is 1: grid size reduction between multigrid levels
    DeprecatedAndRemoved d_merge_local_levels; ///< if merge_method is 1: up to the specified level of the multigrid, constraints are grouped locally, i.e. separately within each contact pairs, while on upper levels they are grouped globally independently of contact pairs.
    Data<type::vector< SReal >> d_constraintForces; ///< OUTPUT: constraint forces (stored only if computeConstraintForces=True)
    Data<bool> d_computeConstraintForces; ///< The indices of the constraintForces to store in the constraintForce data field

    Data < std::set<int> > d_constraintGroups; ///< list of ID of groups of constraints to be handled by this solver.

    Data<std::map < std::string, sofa::type::vector<SReal> > > d_graph; ///< Graph of residuals at each iteration

    DeprecatedAndRemoved d_showLevels; ///< Number of constraint levels to display
    Data<SReal> d_showCellWidth; ///< Distance between each constraint cells
    Data<type::Vec3> d_showTranslation; ///< Position of the first cell
    DeprecatedAndRemoved d_showLevelTranslation; ///< Translation between levels

    ConstraintProblem* getConstraintProblem() override;
    void lockConstraintProblem(sofa::core::objectmodel::BaseObject* from, ConstraintProblem* p1, ConstraintProblem* p2=nullptr) override; ///< Do not use the following LCPs until the next call to this function. This is used to prevent concurrent access to the LCP when using a LCPForceFeedback through an haptic thread

private:
    void addRegularization(linearalgebra::BaseMatrix& W);
    void computeInitialGuess();
    void keepContactForcesValue();

    unsigned int _numConstraints;

    /// Call the method getConstraintInfo on all the BaseConstraintSet
    void getConstraintInfo(core::ConstraintParams cparams);
    /// Call the method addComplianceInConstraintSpace on all the BaseConstraintCorrection
    void addComplianceInConstraintSpace(core::ConstraintParams cparams);

    /// for built lcp ///
    LCPConstraintProblem lcp1, lcp2, lcp3; // Triple buffer for LCP.
    LCPConstraintProblem *current_cp, *last_cp; /// use of last_lcp allows several LCPForceFeedback to be used in the same scene
    sofa::linearalgebra::LPtrFullMatrix<SReal>  *_W;

    /// multi-grid approach ///
    void build_Coarse_Compliance(std::vector<int> &/*constraint_merge*/, int /*sizeCoarseSystem*/);
    sofa::linearalgebra::LPtrFullMatrix<SReal>  _Wcoarse;

    std::vector< int >  m_contact_group;
    std::vector< int >  m_constraint_group;
    std::vector< SReal >  m_constraint_group_fact;
    unsigned int  m_num_group;


    /// common built-unbuilt
    sofa::linearalgebra::FullVector<SReal> *_dFree, *_result;
    void buildSystem();
    ///

    /// for unbuilt lcp ///
    int lcp_gaussseidel_unbuilt(SReal *dfree, SReal *f, std::vector<SReal>* residuals = nullptr);
    int nlcp_gaussseidel_unbuilt(SReal *dfree, SReal *f, std::vector<SReal>* residuals = nullptr);
    int gaussseidel_unbuilt(SReal *dfree, SReal *f, std::vector<SReal>* residuals = nullptr);

    sofa::linearalgebra::SparseMatrix<SReal> _Wdiag;
    std::vector<core::behavior::BaseConstraintCorrection*> _cclist_elem1;
    std::vector<core::behavior::BaseConstraintCorrection*> _cclist_elem2;

    typedef core::behavior::BaseLagrangianConstraint::ConstraintBlockInfo ConstraintBlockInfo;
    typedef core::behavior::BaseLagrangianConstraint::PersistentID PersistentID;

    typedef core::behavior::BaseLagrangianConstraint::VecConstraintBlockInfo VecConstraintBlockInfo;
    typedef core::behavior::BaseLagrangianConstraint::VecPersistentID VecPersistentID;

    class ConstraintBlockBuf
    {
    public:
        std::map<PersistentID,int> persistentToConstraintIdMap;
        int nbLines; ///< how many dofs (i.e. lines in the matrix) are used by each constraint
    };

    std::map<core::behavior::BaseLagrangianConstraint*, ConstraintBlockBuf> _previousConstraints;
    type::vector< SReal > _previousForces;

    VecConstraintBlockInfo  m_constraintBlockInfo;
    VecPersistentID  m_constraintIds;

    // for gaussseidel_unbuilt
    type::vector< helper::LocalBlock33 > unbuilt_W33;
    type::vector< SReal > unbuilt_d;

    type::vector< SReal > unbuilt_W11;

    bool isActive;
};

} //namespace sofa::component::constraint::lagrangian::solver
