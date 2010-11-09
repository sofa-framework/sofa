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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_GENERICLCPCONSTRAINTSOLVER_H
#define SOFA_COMPONENT_CONSTRAINTSET_GENERICLCPCONSTRAINTSOLVER_H

#include <sofa/component/constraintset/LCPConstraintSolver.h>
#include <sofa/core/behavior/BaseConstraint.h>
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/core/behavior/BaseConstraintCorrection.h>

#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/MechanicalVisitor.h>

#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>

#include <sofa/helper/set.h>
#include <sofa/helper/map.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

using namespace sofa::defaulttype;
using namespace sofa::component::linearsolver;
using namespace helper::system::thread;
using core::behavior::ConstraintResolution;
class GenericLCPConstraintSolver;

class SOFA_COMPONENT_CONSTRAINTSET_API GenericLCP : public LCP
{
public:
    FullVector<double> _d, _df;
    std::vector<core::behavior::ConstraintResolution*> constraintsResolutions;
    bool scaleTolerance, allVerified;
    double sor;

    GenericLCP() : scaleTolerance(true), allVerified(false), sor(1.0) {}

    void setMaxConst(unsigned int /*nbC*/) {}
    void setNbConst(unsigned int nbC);
    void gaussSeidel(double timeout = 0, GenericLCPConstraintSolver* solver = NULL);
    void freeConstraintResolutions();
};

class SOFA_COMPONENT_CONSTRAINTSET_API GenericLCPConstraintSolver : public LCPConstraintSolverInterface
{
    typedef std::vector<core::behavior::BaseConstraintCorrection*> list_cc;
    typedef std::vector<list_cc> VecListcc;
    typedef sofa::core::MultiVecId MultiVecId;

public:
    SOFA_CLASS(GenericLCPConstraintSolver, sofa::core::behavior::ConstraintSolver);

    GenericLCPConstraintSolver();
    virtual ~GenericLCPConstraintSolver();

    void init();

    bool prepareStates(double dt, MultiVecId, core::ConstraintParams::ConstOrder = core::ConstraintParams::POS);
    bool buildSystem(double dt, MultiVecId, core::ConstraintParams::ConstOrder = core::ConstraintParams::POS);
    bool solveSystem(double dt, MultiVecId, core::ConstraintParams::ConstOrder = core::ConstraintParams::POS);
    bool applyCorrection(double dt, MultiVecId, core::ConstraintParams::ConstOrder = core::ConstraintParams::POS);

    Data<bool> displayTime;
    Data<int> maxIt;
    Data<double> tolerance, sor;
    Data<bool> scaleTolerance, allVerified, schemeCorrection;
    Data<std::map < std::string, sofa::helper::vector<double> > > graphErrors, graphConstraints /*, graphForces */;

    LCP* getLCP();
    void lockLCP(LCP* l1, LCP* l2=0);

private:

    std::vector<core::behavior::BaseConstraintCorrection*> constraintCorrections;

    void build_LCP();
    GenericLCP lcp1, lcp2, lcp3; // Triple buffer for LCP.
    GenericLCP *lcp,*last_lcp; /// use of last_lcp allows several LCPForceFeedback to be used in the same scene

    simulation::Node *context;

    CTime timer;
    CTime timerTotal;

    double time;
    double timeTotal;
    double timeScale;
};


class SOFA_COMPONENT_CONSTRAINTSET_API MechanicalGetConstraintResolutionVisitor : public simulation::BaseMechanicalVisitor
{
public:
    MechanicalGetConstraintResolutionVisitor(std::vector<core::behavior::ConstraintResolution*>& res, const core::ConstraintParams* params)
        : simulation::BaseMechanicalVisitor(params) , cparams(params)
        , _res(res), _offset(0)
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
            c->getConstraintResolution(_res, _offset);
            end(node, c, t0);
        }
        return RESULT_CONTINUE;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() { }
#endif
private:
    /// Constraint parameters
    const sofa::core::ConstraintParams *cparams;

    std::vector<core::behavior::ConstraintResolution*>& _res;
    unsigned int _offset;
};

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
