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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_LCPCONSTRAINTSOLVER_H
#define SOFA_COMPONENT_CONSTRAINTSET_LCPCONSTRAINTSOLVER_H

#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/core/behavior/BaseConstraintCorrection.h>

#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/MechanicalVisitor.h>

#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>

#include <sofa/helper/set.h>
#include <sofa/helper/map.h>
#include <sofa/helper/LCPcalc.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

using namespace sofa::defaulttype;
using namespace sofa::component::linearsolver;
using namespace helper::system::thread;


/// Christian : WARNING: this class is already defined in sofa::helper
class LCP
{
public:
    int maxConst;
    LPtrFullMatrix<double> W;
    FullVector<double> dFree, f;
    double tol;
    int numItMax;
    unsigned int nbConst;
    bool useInitialF;
    double mu;
    int dim;
private:
    bool lok;

public:
    LCP(unsigned int maxConstraint);
    ~LCP();
    void reset(void);
    //LCP& operator=(LCP& lcp);
    inline double** getW(void) {return W.lptr();};
    inline double& getMu(void) { return mu;};
    inline double* getDfree(void) {return dFree.ptr();};
    inline int getDfreeSize(void) {return dFree.size();};
    inline double getTolerance(void) {return tol;};
    inline void setTol(double t) {tol = t;};
    inline double getMaxIter(void) {return numItMax;};
    inline double* getF(void) {return f.ptr();};
    inline bool useInitialGuess(void) {return useInitialF;};
    inline unsigned int getNbConst(void) {return nbConst;};
    inline void setNbConst(unsigned int nbC) {nbConst = nbC;};
    inline unsigned int getMaxConst(void) {return maxConst;};
    void setMaxConst(unsigned int nbC);

    inline bool isLocked(void) {return false;};
    inline void lock(void) {lok = true;};
    inline void unlock(void) {lok = false;};
    inline void wait(void) {while(lok) ; } //infinite loop?
};



class MechanicalResetContactForceVisitor : public simulation::BaseMechanicalVisitor
{
public:
    //core::MultiVecDerivId force;
    MechanicalResetContactForceVisitor(/*core::MultiVecDerivId force,*/ const core::ExecParams* params)
        : simulation::BaseMechanicalVisitor(params)
        //	, force(force)
    {
    }

    virtual Result fwdMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* ms)
    {
        ctime_t t0 = begin(node, ms);
        ms->resetContactForce(/*force*/);
        end(node, ms, t0);
        return RESULT_CONTINUE;
    }

    virtual Result fwdMappedMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* ms)
    {
        ctime_t t0 = begin(node, ms);
        ms->resetForce(/*force*/);
        end(node, ms, t0);
        return RESULT_CONTINUE;
    }

    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalResetContactForceVisitor";}

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
};


/// Apply the Contact Forces on mechanical models & Compute displacements
class SOFA_COMPONENT_MASTERSOLVER_API MechanicalApplyContactForceVisitor : public simulation::BaseMechanicalVisitor
{
public:
    //VecId force;
    MechanicalApplyContactForceVisitor(double *f, const core::ExecParams* params)
        : simulation::BaseMechanicalVisitor(params)
        , _f(f)
    {
    }

    virtual Result fwdMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* ms)
    {
        ctime_t t0 = begin(node, ms);
        ms->applyContactForce(_f);
        end(node, ms, t0);
        return RESULT_CONTINUE;
    }

    virtual Result fwdMappedMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* ms)
    {
        ctime_t t0 = begin(node, ms);
        ms->applyContactForce(_f);
        end(node, ms, t0);
        return RESULT_CONTINUE;
    }


    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalApplyContactForceVisitor";}

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif

private:
    double *_f; // vector of contact forces from lcp //
    // to be multiplied by constraint direction in mechanical models //

};


/// Gets the vector of constraint values
class MechanicalGetConstraintValueVisitor : public simulation::BaseMechanicalVisitor
{
public:

    MechanicalGetConstraintValueVisitor(BaseVector *v, const core::ConstraintParams* params)
        : simulation::BaseMechanicalVisitor(params)
        , cparams(params)
        , m_v(v)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual Result fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet)
    {
        if (core::behavior::BaseConstraintSet *c=dynamic_cast<core::behavior::BaseConstraintSet*>(cSet))
        {
            ctime_t t0 = begin(node, c);
            c->getConstraintViolation(m_v, cparams);
            end(node, c, t0);
        }
        return RESULT_CONTINUE;
    }

    /// This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    virtual bool stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
    {
        return false; // !map->isMechanical();
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalGetConstraintValueVisitor";}

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif

private:
    /// Constraint parameters
    const sofa::core::ConstraintParams *cparams;

    /// Vector for constraint values
    BaseVector* m_v;
};


class MechanicalGetConstraintInfoVisitor : public simulation::BaseMechanicalVisitor
{
public:
    typedef core::behavior::BaseConstraint::VecConstraintBlockInfo VecConstraintBlockInfo;
    typedef core::behavior::BaseConstraint::VecPersistentID VecPersistentID;
    typedef core::behavior::BaseConstraint::VecConstCoord VecConstCoord;
    typedef core::behavior::BaseConstraint::VecConstDeriv VecConstDeriv;
    typedef core::behavior::BaseConstraint::VecConstArea VecConstArea;

    MechanicalGetConstraintInfoVisitor(VecConstraintBlockInfo& blocks, VecPersistentID& ids, VecConstCoord& positions, VecConstDeriv& directions, VecConstArea& areas, const core::ExecParams* params)
        : simulation::BaseMechanicalVisitor(params)
        , _blocks(blocks)
        , _ids(ids)
        , _positions(positions)
        , _directions(directions)
        , _areas(areas)
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
};


class SOFA_COMPONENT_CONSTRAINTSET_API LCPConstraintSolver : public sofa::core::behavior::ConstraintSolver
{
    typedef std::vector<core::behavior::BaseConstraintCorrection*> list_cc;
    typedef std::vector<list_cc> VecListcc;
    typedef sofa::core::MultiVecId MultiVecId;

public:
    SOFA_CLASS(LCPConstraintSolver, sofa::core::behavior::ConstraintSolver);


    /**
    * @brief Default Constructor
    */
    LCPConstraintSolver();

    /**
    * @brief Default Destructor
    */
    virtual ~LCPConstraintSolver();

    void init();

    bool prepareStates(double dt, MultiVecId, core::ConstraintParams::ConstOrder = core::ConstraintParams::POS);
    bool buildSystem(double dt, MultiVecId, core::ConstraintParams::ConstOrder = core::ConstraintParams::POS);
    bool solveSystem(double dt, MultiVecId, core::ConstraintParams::ConstOrder = core::ConstraintParams::POS);
    bool applyCorrection(double dt, MultiVecId, core::ConstraintParams::ConstOrder = core::ConstraintParams::POS);

    void draw();


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

    Data < helper::set<int> > constraintGroups;

    Data<std::map < std::string, sofa::helper::vector<double> > > f_graph;

    Data<int> showLevels;
    Data<double> showCellWidth;
    Data<defaulttype::Vector3> showTranslation;
    Data<defaulttype::Vector3> showLevelTranslation;

    LCP* getLCP();
    void lockLCP(LCP* l1, LCP* l2=0); ///< Do not use the following LCPs until the next call to this function. This is used to prevent concurent access to the LCP when using a LCPForceFeedback through an haptic thread

private:
    std::vector<core::behavior::BaseConstraintCorrection*> constraintCorrections;
    void computeInitialGuess();
    void keepContactForcesValue();

    unsigned int _numConstraints;
    double _mu;


    /// for built lcp ///
    void build_LCP();
    LCP lcp1, lcp2, lcp3; // Triple buffer for LCP.
    LPtrFullMatrix<double>  *_W;
    LCP *lcp,*last_lcp; /// use of last_lcp allows several LCPForceFeedback to be used in the same scene

    /// multi-grid approach ///
    void MultigridConstraintsMerge();
    void MultigridConstraintsMerge_Compliance();
    void MultigridConstraintsMerge_Spatial();
    void build_Coarse_Compliance(std::vector<int> &/*constraint_merge*/, int /*sizeCoarseSystem*/);
    LPtrFullMatrix<double>  _Wcoarse;

    //std::vector< int> _contact_group;
    //std::vector< int> _constraint_group;
    //std::vector<int> _group_lead;

    std::vector< std::vector< int > > hierarchy_contact_group;
    std::vector< std::vector< int > > hierarchy_constraint_group;
    std::vector< std::vector< double > > hierarchy_constraint_group_fact;
    std::vector< unsigned int > hierarchy_num_group;


    /// common built-unbuilt
    simulation::Node *context;
    FullVector<double> *_dFree, *_result;
    ///
    CTime timer;
    CTime timerTotal;

    double time;
    double timeTotal;
    double timeScale;


    /// for unbuilt lcp ///
    void build_problem_info();
    int lcp_gaussseidel_unbuilt(double *dfree, double *f);
    int nlcp_gaussseidel_unbuilt(double *dfree, double *f);
    int gaussseidel_unbuilt(double *dfree, double *f) { if (_mu == 0.0) return lcp_gaussseidel_unbuilt(dfree, f); else return nlcp_gaussseidel_unbuilt(dfree, f); }

    SparseMatrix<double> *_Wdiag;
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
