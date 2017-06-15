/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_LCPCONSTRAINTSOLVER_H
#define SOFA_COMPONENT_CONSTRAINTSET_LCPCONSTRAINTSOLVER_H
#include "config.h"

#include <SofaConstraint/ConstraintSolverImpl.h>
#include <sofa/core/behavior/BaseConstraintCorrection.h>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/MechanicalVisitor.h>

#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>

#include <sofa/helper/set.h>
#include <sofa/helper/map.h>
#include <sofa/helper/LCPcalc.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

/// Christian : WARNING: this class is already defined in sofa::helper
class LCPConstraintProblem : public ConstraintProblem
{
public:
    double mu;

    void solveTimed(double tolerance, int maxIt, double timeout);
};

class MechanicalGetConstraintInfoVisitor : public simulation::BaseMechanicalVisitor
{
public:
    typedef core::behavior::BaseConstraint::VecConstraintBlockInfo VecConstraintBlockInfo;
    typedef core::behavior::BaseConstraint::VecPersistentID VecPersistentID;
    typedef core::behavior::BaseConstraint::VecConstCoord VecConstCoord;
    typedef core::behavior::BaseConstraint::VecConstDeriv VecConstDeriv;
    typedef core::behavior::BaseConstraint::VecConstArea VecConstArea;

    MechanicalGetConstraintInfoVisitor(const core::ConstraintParams* params, VecConstraintBlockInfo& blocks, VecPersistentID& ids, VecConstCoord& positions, VecConstDeriv& directions, VecConstArea& areas)
        : simulation::BaseMechanicalVisitor(params)
        , _blocks(blocks)
        , _ids(ids)
        , _positions(positions)
        , _directions(directions)
        , _areas(areas)
        , _cparams(params)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual Result fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet)
    {
        if (core::behavior::BaseConstraint *c=cSet->toBaseConstraint())
        {
            ctime_t t0 = begin(node, c);
            c->getConstraintInfo(_cparams, _blocks, _ids, _positions, _directions, _areas);
            end(node, c, t0);
        }
        return RESULT_CONTINUE;
    }


    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalGetConstraintInfoVisitor";}

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
    const core::ConstraintParams* _cparams;
};

class SOFA_CONSTRAINT_API LCPConstraintSolver : public ConstraintSolverImpl
{
public:
    SOFA_CLASS(LCPConstraintSolver, ConstraintSolverImpl);

    typedef std::vector<core::behavior::BaseConstraintCorrection*> list_cc;
    typedef std::vector<list_cc> VecListcc;
    typedef sofa::core::MultiVecId MultiVecId;

protected:
    /**
    * @brief Default Constructor
    */
    LCPConstraintSolver();

    /**
    * @brief Default Destructor
    */
    virtual ~LCPConstraintSolver();
public:
    void init();

    void cleanup();

    bool prepareStates(const core::ConstraintParams * /*cParams*/, MultiVecId res1, MultiVecId res2=MultiVecId::null());
    bool buildSystem(const core::ConstraintParams * /*cParams*/, MultiVecId res1, MultiVecId res2=MultiVecId::null());
    bool solveSystem(const core::ConstraintParams * /*cParams*/, MultiVecId res1, MultiVecId res2=MultiVecId::null());
    bool applyCorrection(const core::ConstraintParams * /*cParams*/, MultiVecId res1, MultiVecId res2=MultiVecId::null());

    void draw(const core::visual::VisualParams* vparams);


    Data<bool> displayDebug;
    Data<bool> displayTime;
    Data<bool> initial_guess;
    Data<bool> build_lcp;
    Data<double> tol;
    Data<int> maxIt;
    Data<double> mu;
    Data<double> minW;
    Data<double> maxF;
    Data<bool> multi_grid;
    Data<int> multi_grid_levels;
    Data<int> merge_method;
    Data<int> merge_spatial_step;
    Data<int> merge_local_levels;

    Data < std::set<int> > constraintGroups;

    Data<std::map < std::string, sofa::helper::vector<double> > > f_graph;

    Data<int> showLevels;
    Data<double> showCellWidth;
    Data<defaulttype::Vector3> showTranslation;
    Data<defaulttype::Vector3> showLevelTranslation;

    ConstraintProblem* getConstraintProblem();
    void lockConstraintProblem(ConstraintProblem* p1, ConstraintProblem* p2=0); ///< Do not use the following LCPs until the next call to this function. This is used to prevent concurent access to the LCP when using a LCPForceFeedback through an haptic thread

    virtual void removeConstraintCorrection(core::behavior::BaseConstraintCorrection *s);

    private:
    std::vector<core::behavior::BaseConstraintCorrection*> constraintCorrections;
	std::vector<char> constraintCorrectionIsActive; // for each constraint correction, a boolean that is false if the parent node is sleeping
    void computeInitialGuess();
    void keepContactForcesValue();

    unsigned int _numConstraints;
    double _mu;

    /// for built lcp ///
    void build_LCP();
    LCPConstraintProblem lcp1, lcp2, lcp3; // Triple buffer for LCP.
    LCPConstraintProblem *lcp, *last_lcp; /// use of last_lcp allows several LCPForceFeedback to be used in the same scene
    sofa::component::linearsolver::LPtrFullMatrix<double>  *_W;

    /// multi-grid approach ///
    void MultigridConstraintsMerge();
    void MultigridConstraintsMerge_Compliance();
    void MultigridConstraintsMerge_Spatial();
    void build_Coarse_Compliance(std::vector<int> &/*constraint_merge*/, int /*sizeCoarseSystem*/);
    sofa::component::linearsolver::LPtrFullMatrix<double>  _Wcoarse;

    //std::vector< int> _contact_group;
    //std::vector< int> _constraint_group;
    //std::vector<int> _group_lead;

    std::vector< std::vector< int > > hierarchy_contact_group;
    std::vector< std::vector< int > > hierarchy_constraint_group;
    std::vector< std::vector< double > > hierarchy_constraint_group_fact;
    std::vector< unsigned int > hierarchy_num_group;


    /// common built-unbuilt
    simulation::Node *context;
    sofa::component::linearsolver::FullVector<double> *_dFree, *_result;
    ///
    sofa::helper::system::thread::CTime timer;
    sofa::helper::system::thread::CTime timerTotal;

    double time;
    double timeTotal;
    double timeScale;


    /// for unbuilt lcp ///
    void build_problem_info();
    int lcp_gaussseidel_unbuilt(double *dfree, double *f, std::vector<double>* residuals = NULL);
    int nlcp_gaussseidel_unbuilt(double *dfree, double *f, std::vector<double>* residuals = NULL);
    int gaussseidel_unbuilt(double *dfree, double *f, std::vector<double>* residuals = NULL) { if (_mu == 0.0) return lcp_gaussseidel_unbuilt(dfree, f, residuals); else return nlcp_gaussseidel_unbuilt(dfree, f, residuals); }

    sofa::component::linearsolver::SparseMatrix<double> *_Wdiag;
    //std::vector<helper::LocalBlock33 *> _Wdiag;
    std::vector<core::behavior::BaseConstraintCorrection*> _cclist_elem1;
    std::vector<core::behavior::BaseConstraintCorrection*> _cclist_elem2;

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

    helper::vector< VecConstraintBlockInfo > hierarchy_constraintBlockInfo;
    helper::vector< VecPersistentID > hierarchy_constraintIds;
    helper::vector< VecConstCoord > hierarchy_constraintPositions;
    helper::vector< VecConstDeriv > hierarchy_constraintDirections;
    helper::vector< VecConstArea > hierarchy_constraintAreas;

    // for gaussseidel_unbuilt
    helper::vector< helper::LocalBlock33 > unbuilt_W33;
    helper::vector< double > unbuilt_d;

    helper::vector< double > unbuilt_W11;
    helper::vector< double > unbuilt_invW11;

    bool isActive;
};

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
