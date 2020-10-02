/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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

#include <SofaConstraint/LCPConstraintSolver.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/SolveVisitor.h>

#include <sofa/simulation/Simulation.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/gl/Cylinder.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/system/thread/CTime.h>
#include <math.h>
#include <iostream>

#include <sofa/core/ObjectFactory.h>

using sofa::core::VecId;

namespace sofa
{

namespace component
{

namespace constraintset
{

void LCPConstraintProblem::solveTimed(double tolerance, int maxIt, double timeout)
{
    helper::nlcp_gaussseidelTimed(dimension, getDfree(), getW(), getF(), mu, tolerance, maxIt, true, timeout);
}

bool LCPConstraintSolver::prepareStates(const core::ConstraintParams * /*cParams*/, MultiVecId /*res1*/, MultiVecId /*res2*/)
{
    sofa::helper::AdvancedTimer::StepVar vtimer("PrepareStates");

    last_lcp = lcp;
    simulation::MechanicalVOpVisitor(core::ExecParams::defaultInstance(), (core::VecId)core::VecDerivId::dx()).setMapped(true).execute( context); //dX=0

    msg_info() <<" propagate DXn performed - collision called" ;

    time = 0.0;
    timeTotal=0.0;
    timeScale = 1000.0 / (double)sofa::helper::system::thread::CTime::getTicksPerSec();

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }

    if ( displayTime.getValue() )
    {
        time = (double) timer.getTime();
        timeTotal = (double) timerTotal.getTime();
    }
    return true;
}

bool LCPConstraintSolver::buildSystem(const core::ConstraintParams * /*cParams*/, MultiVecId /*res1*/, MultiVecId /*res2*/)
{
    //sout<<"constraintCorrections is called"<<sendl;

    // Test if the nodes containing the constraint correction are active (not sleeping)
    for (unsigned int i = 0; i < constraintCorrections.size(); i++)
        constraintCorrectionIsActive[i] = !constraintCorrections[i]->getContext()->isSleeping();

    if(build_lcp.getValue())
    {
        //sout<<"build_LCP is called"<<sendl;
        build_LCP();

        if ( displayTime.getValue() )
        {
            sout<<" build_LCP " << ( (double) timer.getTime() - time)*timeScale<<" ms" <<sendl;
            time = (double) timer.getTime();
        }
    }
    else
    {
        build_problem_info();
        //std::cout<<"build_problem_info is finished"<<std::endl;
        if ( displayTime.getValue() )
        {
            sout<<" build_problem " << ( (double) timer.getTime() - time)*timeScale<<" ms" <<sendl;
            time = (double) timer.getTime();
        }
    }
    return true;
}

bool LCPConstraintSolver::solveSystem(const core::ConstraintParams * /*cParams*/, MultiVecId /*res1*/, MultiVecId /*res2*/)
{

    std::map < std::string, sofa::helper::vector<double> >& graph = *f_graph.beginEdit();

    if (build_lcp.getValue())
    {

        double _tol = tol.getValue();
        int _maxIt = maxIt.getValue();
        double _minW = minW.getValue();
        double _maxF = maxF.getValue();

        if (_mu > 0.0)
        {
            lcp->tolerance = _tol;

            if (multi_grid.getValue())
            {
                //std::cout<<"+++++++++++++ \n SOLVE WITH MULTIGRID \n ++++++++++++++++"<<std::endl;

                sofa::helper::AdvancedTimer::stepBegin("ConstraintsMerge");
                MultigridConstraintsMerge();
                sofa::helper::AdvancedTimer::stepEnd  ("ConstraintsMerge");
                //build_Coarse_Compliance(_constraint_group, 3*_group_lead.size());
                //msg_info()<<"out from build_Coarse_Compliance"<<std::endl;

                sofa::helper::vector<double>& graph_residuals = graph["Error"];
                graph_residuals.clear();
                sofa::helper::vector<double>& graph_violations = graph["Violation"];
                graph_violations.clear();
                sofa::helper::vector<double>& graph_levels = graph["Level"];
                graph_levels.clear();

                /*helper::nlcp_multiGrid(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _mu, _tol, _maxIt, initial_guess.getValue(),
                _Wcoarse.lptr(),
                _contact_group, _group_lead.size(), notMuted());*/


                sofa::helper::AdvancedTimer::stepBegin("NLCP MultiGrid");
                helper::nlcp_multiGrid_Nlevels(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _mu, _tol, _maxIt, initial_guess.getValue(),
                        hierarchy_contact_group, hierarchy_num_group, hierarchy_constraint_group, hierarchy_constraint_group_fact,  notMuted(), &graph_residuals, &graph_levels, &graph_violations);
                sofa::helper::AdvancedTimer::stepEnd("NLCP MultiGrid");

                //helper::nlcp_multiGrid_2levels(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _mu, _tol, _maxIt, initial_guess.getValue(),
                //                       _contact_group, _group_lead.size(),  notMuted(), &graph_residuals, &graph_levels);
                //std::cout<<"+++++++++++++ \n SOLVE WITH GAUSSSEIDEL \n ++++++++++++++++"<<std::endl;
                //helper::nlcp_gaussseidel(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _mu, _tol, _maxIt, initial_guess.getValue(),
                //                         notMuted(), &graph_residuals);

                // if ( notMuted()) helper::afficheLCP(_dFree->ptr(), _W->lptr(), _result->ptr(),_numConstraints);
            }
            else
            {
                sofa::helper::vector<double>& graph_error = graph["Error"];
                graph_error.clear();
                sofa::helper::vector<double>& graph_violations = graph["Violation"];
                graph_violations.clear();
                sofa::helper::AdvancedTimer::stepBegin("NLCP GaussSeidel");
                helper::nlcp_gaussseidel(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _mu, _tol, _maxIt, initial_guess.getValue(),
                        notMuted(), _minW, _maxF, &graph_error, &graph_violations);
                sofa::helper::AdvancedTimer::stepEnd("NLCP GaussSeidel");
             }
        }
        else
        {
            sofa::helper::vector<double>& graph_error = graph["Error"];
            graph_error.clear();
            sofa::helper::AdvancedTimer::stepBegin("LCP GaussSeidel");
            helper::gaussSeidelLCP1(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _tol, _maxIt, _minW, _maxF, &graph_error);
            sofa::helper::AdvancedTimer::stepEnd  ("LCP GaussSeidel");
            if (notMuted()) helper::afficheLCP(_dFree->ptr(), _W->lptr(), _result->ptr(),_numConstraints);
        }
    }
    else
    {

        sofa::helper::vector<double>& graph_error = graph["Error"];
        graph_error.clear();
        sofa::helper::AdvancedTimer::stepBegin("NLCP GaussSeidel Unbuild");
        gaussseidel_unbuilt(_dFree->ptr(), _result->ptr(), &graph_error);
        sofa::helper::AdvancedTimer::stepBegin("NLCP GaussSeidel Unbuild");

        if (displayDebug.getValue())
        {
            dmsg_info() <<"_result unbuilt:"<<(*_result) ;

            /////// debug
            _result->resize(_numConstraints);

            double _tol = tol.getValue();
            int _maxIt = maxIt.getValue();

            build_LCP();

            helper::nlcp_gaussseidel(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _mu, _tol, _maxIt, initial_guess.getValue());
            dmsg_info() <<"\n_result nlcp :"<<(*_result);
        }

        ////////
    }

    if ( displayTime.getValue() )
    {
        msg_info() <<" TOTAL solve_LCP " <<( (double) timer.getTime() - time)*timeScale<<" ms" ;
        time = (double) timer.getTime();
    }

    f_graph.endEdit();
    return true;
}

bool LCPConstraintSolver::applyCorrection(const core::ConstraintParams * /*cParams*/, MultiVecId /*res1*/, MultiVecId /*res2*/)
{
    if (initial_guess.getValue())
        keepContactForcesValue();

    dmsg_info() << "keepContactForces done" ;

    sofa::helper::AdvancedTimer::stepBegin("Apply Contact Force");

    for (unsigned int i = 0; i < constraintCorrections.size(); i++)
    {
        if (!constraintCorrectionIsActive[i]) continue;
        core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->applyContactForce(_result);
    }
    sofa::helper::AdvancedTimer::stepEnd  ("Apply Contact Force");

    dmsg_info() <<"applyContactForce in constraintCorrection done" ;

    dmsg_info() <<" TotalTime "
                <<( (double) timerTotal.getTime() - timeTotal)*timeScale <<" ms" ;

    return true;
}

#define MAX_NUM_CONSTRAINTS 3000
//#define DISPLAY_TIME

LCPConstraintSolver::LCPConstraintSolver()
    : displayDebug(initData(&displayDebug, false, "displayDebug","Display debug information."))
    , displayTime(initData(&displayTime, false, "displayTime","Display time for each important step of LCPConstraintSolver."))
    , initial_guess(initData(&initial_guess, true, "initial_guess","activate LCP results history to improve its resolution performances."))
    , build_lcp(initData(&build_lcp, true, "build_lcp", "LCP is not fully built to increase performance in some case."))
    , tol( initData(&tol, 0.001, "tolerance", "residual error threshold for termination of the Gauss-Seidel algorithm"))
    , maxIt( initData(&maxIt, 1000, "maxIt", "maximal number of iterations of the Gauss-Seidel algorithm"))
    , mu( initData(&mu, 0.6, "mu", "Friction coefficient"))
    , minW( initData(&minW, 0.0, "minW", "If not zero, constraints whose self-compliance (i.e. the corresponding value on the diagonal of W) is smaller than this threshold will be ignored"))
    , maxF( initData(&maxF, 0.0, "maxF", "If not zero, constraints whose response force becomes larger than this threshold will be ignored"))
    , multi_grid(initData(&multi_grid, false, "multi_grid","activate multi_grid resolution (NOT STABLE YET)"))
    , multi_grid_levels(initData(&multi_grid_levels, 2, "multi_grid_levels","if multi_grid is active: how many levels to create (>=2)"))
    , merge_method( initData(&merge_method, 0, "merge_method","if multi_grid is active: which method to use to merge constraints (0 = compliance-based, 1 = spatial coordinates)"))
    , merge_spatial_step( initData(&merge_spatial_step, 2, "merge_spatial_step", "if merge_method is 1: grid size reduction between multigrid levels"))
    , merge_local_levels( initData(&merge_local_levels, 2, "merge_local_levels", "if merge_method is 1: up to the specified level of the multigrid, constraints are grouped locally, i.e. separately within each contact pairs, while on upper levels they are grouped globally independently of contact pairs."))
    , constraintGroups( initData(&constraintGroups, "group", "list of ID of groups of constraints to be handled by this solver."))
    , f_graph( initData(&f_graph,"graph","Graph of residuals at each iteration"))
    , showLevels( initData(&showLevels,0,"showLevels","Number of constraint levels to display"))
    , showCellWidth( initData(&showCellWidth, "showCellWidth", "Distance between each constraint cells"))
    , showTranslation( initData(&showTranslation, "showTranslation", "Position of the first cell"))
    , showLevelTranslation( initData(&showLevelTranslation, "showLevelTranslation", "Translation between levels"))
    , _mu(0.6)
    , lcp(&lcp1)
    , last_lcp(0)
    , _W(&lcp1.W)
    , _dFree(&lcp1.dFree)
    , _result(&lcp1.f)
    , _Wdiag(NULL)
{
    _numConstraints = 0;
    _mu = 0.0;
    constraintGroups.beginEdit()->insert(0);
    constraintGroups.endEdit();

    f_graph.setWidget("graph");
    //f_graph.setReadOnly(true);

    //_numPreviousContact=0;
    //_PreviousContactList = (contactBuf *)malloc(MAX_NUM_CONSTRAINTS * sizeof(contactBuf));
    //_cont_id_list = (long *)malloc(MAX_NUM_CONSTRAINTS * sizeof(long));

    _Wdiag = new sofa::component::linearsolver::SparseMatrix<double>();

    tol.setRequired(true);
    maxIt.setRequired(true);
}

LCPConstraintSolver::~LCPConstraintSolver()
{
    if (_Wdiag != 0)
        delete _Wdiag;
}

void LCPConstraintSolver::init()
{
    core::behavior::ConstraintSolver::init();

    // Prevents ConstraintCorrection accumulation due to multiple AnimationLoop initialization on dynamic components Add/Remove operations.
    if (!constraintCorrections.empty())
    {
        for (unsigned int i = 0; i < constraintCorrections.size(); i++)
            constraintCorrections[i]->removeConstraintSolver(this);
        constraintCorrections.clear();
    }

    getContext()->get<core::behavior::BaseConstraintCorrection>(&constraintCorrections, core::objectmodel::BaseContext::SearchDown);
    constraintCorrectionIsActive.resize(constraintCorrections.size());
    for (unsigned int i = 0; i < constraintCorrections.size(); i++)
        constraintCorrections[i]->addConstraintSolver(this);

    context = (simulation::Node*) getContext();
}

void LCPConstraintSolver::cleanup()
{
    if (!constraintCorrections.empty())
    {
        for (unsigned int i = 0; i < constraintCorrections.size(); i++)
            constraintCorrections[i]->removeConstraintSolver(this);
        constraintCorrections.clear();
    }

    core::behavior::ConstraintSolver::cleanup();
}

void LCPConstraintSolver::removeConstraintCorrection(core::behavior::BaseConstraintCorrection *s)
{
    constraintCorrections.erase(std::remove(constraintCorrections.begin(), constraintCorrections.end(), s), constraintCorrections.end());
}

void LCPConstraintSolver::build_LCP()
{
    _numConstraints = 0;
    core::ConstraintParams cparams;

    cparams.setX(core::ConstVecCoordId::freePosition());
    cparams.setV(core::ConstVecDerivId::freeVelocity());

    sofa::helper::AdvancedTimer::stepBegin("Accumulate Constraint");
    // mechanical action executed from root node to propagate the constraints
    simulation::MechanicalResetConstraintVisitor(&cparams).execute(context);
    simulation::MechanicalAccumulateConstraint(&cparams, core::MatrixDerivId::holonomicC(), _numConstraints).execute(context);
    sofa::helper::AdvancedTimer::stepEnd  ("Accumulate Constraint");
    _mu = mu.getValue();
    sofa::helper::AdvancedTimer::valSet("numConstraints", _numConstraints);

    lcp->mu = _mu;
    lcp->clear(_numConstraints);

    sofa::helper::AdvancedTimer::stepBegin("Get Constraint Value");
    MechanicalGetConstraintViolationVisitor(&cparams, _dFree).execute(context);
    sofa::helper::AdvancedTimer::stepEnd("Get Constraint Value");

    dmsg_info() <<"LCPConstraintSolver: "<<_numConstraints<<" constraints, mu = "<<_mu ;

    sofa::helper::AdvancedTimer::stepBegin("Get Compliance");

    dmsg_info() <<" computeCompliance in "  << constraintCorrections.size()<< " constraintCorrections" ;

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->addComplianceInConstraintSpace(&cparams, _W);
    }

    dmsg_info() << "W=" << *_W ;

    sofa::helper::AdvancedTimer::stepEnd  ("Get Compliance");

    dmsg_info() <<" computeCompliance_done " ;

    int nLevels = 1;
    if (multi_grid.getValue())
    {
        nLevels = multi_grid_levels.getValue();
        if (nLevels < 2) nLevels = 2;
    }
    hierarchy_constraintBlockInfo.resize(nLevels);
    hierarchy_constraintIds.resize(nLevels);
    hierarchy_constraintPositions.resize(nLevels);
    hierarchy_constraintDirections.resize(nLevels);
    hierarchy_constraintAreas.resize(nLevels);
    for (int l=0; l<nLevels; ++l)
    {
        hierarchy_constraintBlockInfo[l].clear();
        hierarchy_constraintIds[l].clear();
        hierarchy_constraintPositions[l].clear();
        hierarchy_constraintDirections[l].clear();
        hierarchy_constraintAreas[l].clear();
    }

    if ((initial_guess.getValue() || multi_grid.getValue() || showLevels.getValue()) && (_numConstraints != 0))
    {
        sofa::helper::AdvancedTimer::stepBegin("Get Constraint Info");
        MechanicalGetConstraintInfoVisitor(&cparams, hierarchy_constraintBlockInfo[0], hierarchy_constraintIds[0], hierarchy_constraintPositions[0], hierarchy_constraintDirections[0], hierarchy_constraintAreas[0]).execute(context);
        sofa::helper::AdvancedTimer::stepEnd  ("Get Constraint Info");
        if (initial_guess.getValue())
            computeInitialGuess();
    }
}

void LCPConstraintSolver::build_Coarse_Compliance(std::vector<int> &constraint_merge, int sizeCoarseSystem)
{
    /* constraint_merge => tableau donne l'indice du groupe de contraintes dans le système grossier en fonction de l'indice de la contrainte dans le système de départ */
    dmsg_info() <<"build_Coarse_Compliance is called : size="<<sizeCoarseSystem ;

    _Wcoarse.clear();

    dmsg_error_when(sizeCoarseSystem==0) <<"no constraint" ;

    _Wcoarse.resize(sizeCoarseSystem,sizeCoarseSystem);
    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->getComplianceWithConstraintMerge(&_Wcoarse, constraint_merge);
    }
}

void LCPConstraintSolver::MultigridConstraintsMerge()
{
    switch(merge_method.getValue())
    {
    case 0:
        MultigridConstraintsMerge_Compliance();
        break;
    case 1:
        MultigridConstraintsMerge_Spatial();
        break;
    default:
        serr << "Unsupported merge method " << merge_method.getValue() << sendl;
    }
}

void LCPConstraintSolver::MultigridConstraintsMerge_Compliance()
{
    /////// Analyse des contacts �  regrouper //////
    double criterion=0.0;
    int numContacts = _numConstraints/3;

    hierarchy_contact_group.resize(1);
    hierarchy_constraint_group.resize(1);
    hierarchy_constraint_group_fact.resize(1);
    hierarchy_num_group.resize(1);
    std::vector<int> group_lead;
    std::vector<int>& contact_group = hierarchy_contact_group[0];
    std::vector<int>& constraint_group = hierarchy_constraint_group[0];
    std::vector<double>& constraint_group_fact = hierarchy_constraint_group_fact[0];
    unsigned int& num_group = hierarchy_num_group[0];
    contact_group.clear();
    contact_group.resize(numContacts);
    group_lead.clear();
    constraint_group.clear();
    constraint_group.resize(_numConstraints);
    constraint_group_fact.clear();
    constraint_group_fact.resize(_numConstraints);

    for (int c=0; c<numContacts; c++)
    {
        bool new_group = true;
        for(int g=0; g<(int)group_lead.size() ; g++)
        {
            if (_W->lptr()[3*c][3*group_lead[g]] > criterion * (_W->lptr()[3*c][3*c] +_W->lptr()[3*group_lead[g]][3*group_lead[g]]) )  // on regarde les couplages selon la normale...
            {
                new_group =false;
                contact_group[c] = g;
            }
        }
        if (new_group)
        {
            contact_group[c]=group_lead.size();
            group_lead.push_back(c);
        }
    }
    num_group = group_lead.size();
    dmsg_info() << "contacts merged in "<<num_group<<" list(s)" ;

    for (int c=0; c<numContacts; c++)
    {
        constraint_group[3*c  ] = 3*contact_group[c]  ; constraint_group_fact[3*c  ] = 1.0;
        constraint_group[3*c+1] = 3*contact_group[c]+1; constraint_group_fact[3*c+1] = 1.0;
        constraint_group[3*c+2] = 3*contact_group[c]+2; constraint_group_fact[3*c+2] = 1.0;
    }
}

void LCPConstraintSolver::MultigridConstraintsMerge_Spatial()
{
    const int merge_spatial_step = this->merge_spatial_step.getValue();
    const int merge_spatial_shift = 0; // merge_spatial_step/2
    const int merge_local_levels = this->merge_local_levels.getValue();
    int numConstraints = _numConstraints;
    int numContacts = numConstraints/3;
    int nLevels = multi_grid_levels.getValue();
    if (nLevels < 2) nLevels = 2;

    sout << "Multigrid merge from " << numContacts << " contacts." << sendl;

    hierarchy_contact_group.resize(nLevels-1);
    hierarchy_constraint_group.resize(nLevels-1);
    hierarchy_constraint_group_fact.resize(nLevels-1);
    hierarchy_num_group.resize(nLevels-1);

    hierarchy_constraintBlockInfo.resize(nLevels);
    hierarchy_constraintPositions.resize(nLevels);
    hierarchy_constraintDirections.resize(nLevels);
    hierarchy_constraintAreas.resize(nLevels);

    for (int level = 1; level < nLevels; ++level)
    {
        std::vector<int>& contact_group = hierarchy_contact_group[level-1];
        std::vector<int>& constraint_group = hierarchy_constraint_group[level-1];
        std::vector<double>& constraint_group_fact = hierarchy_constraint_group_fact[level-1];
        unsigned int& num_group = hierarchy_num_group[level-1];

        contact_group.clear();
        contact_group.resize(numContacts);
        constraint_group.clear();
        constraint_group.resize(numConstraints);
        constraint_group_fact.clear();
        constraint_group_fact.resize(numConstraints);
        num_group = 0;

        const VecConstraintBlockInfo& constraintBlockInfo = hierarchy_constraintBlockInfo[level-1];
        const VecConstCoord&          constraintPositions = hierarchy_constraintPositions[level-1];
        const VecConstDeriv&          constraintDirections = hierarchy_constraintDirections[level-1];
        const VecConstArea&           constraintAreas = hierarchy_constraintAreas[level-1];

        VecConstraintBlockInfo& newConstraintBlockInfo = hierarchy_constraintBlockInfo[level];
        VecConstCoord&          newConstraintPositions = hierarchy_constraintPositions[level];
        VecConstDeriv&          newConstraintDirections = hierarchy_constraintDirections[level];
        VecConstArea&           newConstraintAreas = hierarchy_constraintAreas[level];

        newConstraintBlockInfo.clear();
        newConstraintPositions.clear();
        newConstraintDirections.clear();
        newConstraintAreas.clear();

        std::map<ConstCoord, int> coord2coarseId;

        for (unsigned cb = 0; cb < constraintBlockInfo.size(); ++cb)
        {
            const ConstraintBlockInfo& info = constraintBlockInfo[cb];
            sout << "MultigridConstraintsMerge_Spatial level " << level-1 << " constraint block " << cb << " from " << (info.parent ? info.parent->getName() : std::string("NULL"))
                    << " : c0 = " << info.const0 << " nbl = " << info.nbLines << " nbg = " << info.nbGroups << " offsetPosition = " << info.offsetPosition << " offsetDirection = " << info.offsetDirection << " offsetArea = " << info.offsetArea << sendl;
            if (!info.hasPosition)
            {
                serr << "MultigridConstraintsMerge_Spatial: constraints from " << (info.parent ? info.parent->getName() : std::string("NULL")) << " have no position data" << sendl;
                continue;
            }
            if (!info.hasDirection)
            {
                serr << "MultigridConstraintsMerge_Spatial: constraints from " << (info.parent ? info.parent->getName() : std::string("NULL")) << " have no direction data" << sendl;
                continue;
            }
            ConstraintBlockInfo newInfo;
            newInfo = info;
            newInfo.hasArea = true;
            newInfo.offsetPosition = newConstraintPositions.size();
            newInfo.offsetDirection = newConstraintDirections.size();
            newInfo.offsetArea = newConstraintAreas.size();
            newInfo.const0 = num_group * 3;
            const int c0 = info.const0;
            const int nbl = info.nbLines;
            for (int c = 0; c < info.nbGroups; ++c)
            {
                int idFine = c0 + c*nbl;
                if (idFine + 2 >= numConstraints)
                {
                    serr << "MultigridConstraintsMerge_Spatial level " << level << ": constraint " << idFine << " from " << (info.parent ? info.parent->getName() : std::string("NULL")) << " has invalid index" << sendl;
                    break;
                }
                if ((unsigned)(info.offsetPosition + c) >= constraintPositions.size())
                {
                    serr << "MultigridConstraintsMerge_Spatial level " << level << ": constraint " << idFine << " from " << (info.parent ? info.parent->getName() : std::string("NULL")) << " has invalid position index" << sendl;
                    break;
                }
                ConstCoord posFine = constraintPositions[info.offsetPosition + c];
                ConstDeriv dirFineN  = constraintDirections[info.offsetDirection + 3*c + 0];
                ConstDeriv dirFineT1 = constraintDirections[info.offsetDirection + 3*c + 1];
                ConstDeriv dirFineT2 = constraintDirections[info.offsetDirection + 3*c + 2];
                ConstArea area = (info.hasArea) ? constraintAreas[info.offsetArea + c] : (ConstArea)1.0;
                ConstCoord posCoarse;
                for (int i=0; i<3; ++i)
                {
                    int p = posFine[i]+merge_spatial_shift;
                    if (p < 0)
                        p -= merge_spatial_step-1;
                    p = p / merge_spatial_step;
                    posCoarse[i] = p;
                }
                std::pair< std::map<ConstCoord,int>::iterator, bool > res = coord2coarseId.insert(std::map<ConstCoord,int>::value_type(posCoarse, (int)num_group));
                int idCoarse = res.first->second * 3;
                if (res.second)
                {
                    // new group
                    newConstraintPositions.push_back(posCoarse);
                    newConstraintDirections.push_back(dirFineN*area);
                    newConstraintDirections.push_back(dirFineT1*area);
                    newConstraintDirections.push_back(dirFineT2*area);
                    newConstraintAreas.push_back(area);
                    ++num_group;
                }
                else
                {
                    // add to existing group
                    newConstraintAreas[idCoarse/3] += area;
                    ConstDeriv& dirCoarseN  = newConstraintDirections[idCoarse+0];
                    ConstDeriv& dirCoarseT1 = newConstraintDirections[idCoarse+1];
                    ConstDeriv& dirCoarseT2 = newConstraintDirections[idCoarse+2];
                    double dotNN   = dirCoarseN  * dirFineN;
                    double dotT1T1 = dirCoarseT1 * dirFineT1;
                    double dotT2T2 = dirCoarseT2 * dirFineT2;
                    double dotT2T1 = dirCoarseT2 * dirFineT1;
                    double dotT1T2 = dirCoarseT1 * dirFineT2;
                    dirCoarseN  += dirFineN  * ((dotNN < 0) ? -area : area);
                    if (fabs(dotT1T1) + fabs(dotT2T2) > fabs(dotT1T2) + fabs(dotT2T1))
                    {
                        // friction axes are aligned
                        dirCoarseT1 += dirFineT1 * ((dotT1T1 < 0) ? -area : area);
                        dirCoarseT2 += dirFineT2 * ((dotT2T2 < 0) ? -area : area);
                    }
                    else
                    {
                        // friction axes are swapped
                        dirCoarseT1 += dirFineT2 * ((dotT1T2 < 0) ? -area : area);
                        dirCoarseT2 += dirFineT1 * ((dotT2T1 < 0) ? -area : area);
                    }
                }
                contact_group[idFine/3] = idCoarse/3;
                //constraint_group[idFine+0] = idCoarse+0;  constraint_group_fact[idFine+0] = 1.0;
                //constraint_group[idFine+1] = idCoarse+1;  constraint_group_fact[idFine+1] = 1.0;
                //constraint_group[idFine+2] = idCoarse+2;  constraint_group_fact[idFine+2] = 1.0;
            }
            newInfo.nbGroups = num_group - newInfo.const0 / 3;
            newConstraintBlockInfo.push_back(newInfo);
            if (level < merge_local_levels)
            {
                // the following line clears the coarse group map between blocks
                // of constraints, hence disallowing any merging of constraints
                // not created by the same BaseConstraint component
                coord2coarseId.clear();
            }
        }
        // Finalize
        sout << "Multigrid merge level " << level << ": " << num_group << " groups." << sendl;

        // Normalize and orthogonalize constraint directions
        for (unsigned int g=0; g<num_group; ++g)
        {
            int idCoarse = g*3;
            ConstDeriv& dirCoarseN  = newConstraintDirections[idCoarse+0];
            ConstDeriv& dirCoarseT1 = newConstraintDirections[idCoarse+1];
            ConstDeriv& dirCoarseT2 = newConstraintDirections[idCoarse+2];
            dirCoarseT2 = dirCoarseN.cross(dirCoarseT1);
            dirCoarseT1 = dirCoarseT2.cross(dirCoarseN);
            dirCoarseN.normalize();
            dirCoarseT1.normalize();
            dirCoarseT2.normalize();
        }

        // Compute final constraint associations, accounting for possible friction axis flips and swaps
        for (int c=0; c<numContacts; ++c)
        {
            int g = contact_group[c];
            int idFine = c*3;
            int idCoarse = g*3;

            ConstDeriv dirFineN  = constraintDirections[idFine+0];
            ConstDeriv dirFineT1 = constraintDirections[idFine+1];
            ConstDeriv dirFineT2 = constraintDirections[idFine+2];
            ConstDeriv& dirCoarseN  = newConstraintDirections[idCoarse+0];
            ConstDeriv& dirCoarseT1 = newConstraintDirections[idCoarse+1];
            ConstDeriv& dirCoarseT2 = newConstraintDirections[idCoarse+2];
            double dotNN   = dirCoarseN  * dirFineN;
            if (dotNN < 0)
            {
                // constraint direction is flipped, so relative velocities for friction are reversed
                dirFineT1 = -dirFineT1;
                dirFineT2 = -dirFineT2;
            }

            double dotT1T1 = dirCoarseT1 * dirFineT1;
            double dotT2T2 = dirCoarseT2 * dirFineT2;
            double dotT2T1 = dirCoarseT2 * dirFineT1;
            double dotT1T2 = dirCoarseT1 * dirFineT2;
            constraint_group[idFine+0] = idCoarse+0;  constraint_group_fact[idFine+0] = 1.0;

            if (fabs(dotT1T1) + fabs(dotT2T2) > fabs(dotT1T2) + fabs(dotT2T1))
            {
                // friction axes are aligned
                constraint_group[idFine+1] = idCoarse+1;  constraint_group_fact[idFine+1] = ((dotT1T1 < 0) ? -1.0 : 1.0);
                constraint_group[idFine+2] = idCoarse+2;  constraint_group_fact[idFine+2] = ((dotT2T2 < 0) ? -1.0 : 1.0);
            }
            else
            {
                // friction axes are swapped
                constraint_group[idFine+1] = idCoarse+2;  constraint_group_fact[idFine+1] = ((dotT2T1 < 0) ? -1.0 : 1.0);
                constraint_group[idFine+2] = idCoarse+1;  constraint_group_fact[idFine+2] = ((dotT1T2 < 0) ? -1.0 : 1.0);
            }
        }

        numContacts = num_group;
        numConstraints = numContacts*3;
    }
    const VecConstraintBlockInfo& constraintBlockInfo = hierarchy_constraintBlockInfo[nLevels-1];
    for (unsigned cb = 0; cb < constraintBlockInfo.size(); ++cb)
    {
        const ConstraintBlockInfo& info = constraintBlockInfo[cb];
        sout << "MultigridConstraintsMerge_Spatial level " << nLevels-1 << " constraint block " << cb << " from " << (info.parent ? info.parent->getName() : std::string("NULL"))
                << " : c0 = " << info.const0 << " nbl = " << info.nbLines << " nbg = " << info.nbGroups << " offsetPosition = " << info.offsetPosition << " offsetDirection = " << info.offsetDirection << " offsetArea = " << info.offsetArea << sendl;
    }
}


/// build_problem_info
/// When the LCP or the NLCP is not fully built, the  diagonal blocks of the matrix are still needed for the resolution
/// This function ask to the constraintCorrection classes to build this diagonal blocks
void LCPConstraintSolver::build_problem_info()
{
    core::ConstraintParams cparams;

    cparams.setX(core::ConstVecCoordId::freePosition());
    cparams.setV(core::ConstVecDerivId::freeVelocity());

    _numConstraints = 0;

    sofa::helper::AdvancedTimer::stepBegin("Accumulate Constraint");

    // Accumulate Constraints

    simulation::MechanicalResetConstraintVisitor resetCtr(&cparams);
    resetCtr.execute(context);
    simulation::MechanicalAccumulateConstraint accCtr(&cparams, core::MatrixDerivId::holonomicC(), _numConstraints );
    accCtr.execute(context);
    sofa::helper::AdvancedTimer::stepEnd  ("Accumulate Constraint");
    _mu = mu.getValue();
    sofa::helper::AdvancedTimer::valSet("numConstraints", _numConstraints);

    lcp->mu = _mu;
    lcp->clear(_numConstraints);

    // as _Wdiag is a sparse matrix resize do not allocate memory
    _Wdiag->resize(_numConstraints,_numConstraints);

    // debug
    //std::cout<<" resize done "  <<std::endl;

    sofa::helper::AdvancedTimer::stepBegin("Get Constraint Value");
    MechanicalGetConstraintViolationVisitor(&cparams, _dFree).execute(context);
    sofa::helper::AdvancedTimer::stepEnd  ("Get Constraint Value");

    dmsg_info() <<"LCPConstraintSolver: "<<_numConstraints<<" constraints, mu = "<<_mu;

    int nLevels = 1;
    if (multi_grid.getValue())
    {
        nLevels = multi_grid_levels.getValue();
        if (nLevels < 2) nLevels = 2;
    }
    hierarchy_constraintBlockInfo.resize(nLevels);
    hierarchy_constraintIds.resize(nLevels);
    hierarchy_constraintPositions.resize(nLevels);
    hierarchy_constraintDirections.resize(nLevels);
    hierarchy_constraintAreas.resize(nLevels);
    for (int l=0; l<nLevels; ++l)
    {
        hierarchy_constraintBlockInfo[l].clear();
        hierarchy_constraintIds[l].clear();
        hierarchy_constraintPositions[l].clear();
        hierarchy_constraintDirections[l].clear();
        hierarchy_constraintAreas[l].clear();
    }

    if ((initial_guess.getValue() || multi_grid.getValue() || showLevels.getValue()) && (_numConstraints != 0))
    {
        sofa::helper::AdvancedTimer::stepBegin("Get Constraint Info");
        MechanicalGetConstraintInfoVisitor(&cparams, hierarchy_constraintBlockInfo[0], hierarchy_constraintIds[0], hierarchy_constraintPositions[0], hierarchy_constraintDirections[0], hierarchy_constraintAreas[0]).execute(context);
        sofa::helper::AdvancedTimer::stepEnd  ("Get Constraint Info");
        if (initial_guess.getValue())
            computeInitialGuess();
    }
}

void LCPConstraintSolver::computeInitialGuess()
{
    sofa::helper::AdvancedTimer::StepVar vtimer("InitialGuess");

    const VecConstraintBlockInfo& constraintBlockInfo = hierarchy_constraintBlockInfo[0];
    const VecPersistentID& constraintIds = hierarchy_constraintIds[0];
    int numContact = (_mu > 0.0) ? _numConstraints/3 : _numConstraints;

    for (int c=0; c<numContact; c++)
    {
        if (_mu>0.0)
        {
            (*_result)[3*c  ] = 0.0;
            (*_result)[3*c+1] = 0.0;
            (*_result)[3*c+2] = 0.0;
        }
        else
        {
            (*_result)[c] =  0.0;
            (*_result)[c+numContact] =  0.0;
        }
    }
    for (unsigned cb = 0; cb < constraintBlockInfo.size(); ++cb)
    {
        const ConstraintBlockInfo& info = constraintBlockInfo[cb];
        if (!info.hasId) continue;
        std::map<core::behavior::BaseConstraint*, ConstraintBlockBuf>::const_iterator previt = _previousConstraints.find(info.parent);
        if (previt == _previousConstraints.end()) continue;
        const ConstraintBlockBuf& buf = previt->second;
        const int c0 = info.const0;
        const int nbl = (info.nbLines < buf.nbLines) ? info.nbLines : buf.nbLines;
        for (int c = 0; c < info.nbGroups; ++c)
        {
            std::map<PersistentID,int>::const_iterator it = buf.persistentToConstraintIdMap.find(constraintIds[info.offsetId + c]);
            if (it == buf.persistentToConstraintIdMap.end()) continue;
            int prevIndex = it->second;
            if (prevIndex >= 0 && prevIndex+nbl <= (int) _previousForces.size())
            {
                for (int l=0; l<nbl; ++l)
                    (*_result)[c0 + c*nbl + l] = _previousForces[prevIndex + l];
            }
        }
    }
}

void LCPConstraintSolver::keepContactForcesValue()
{
    sofa::helper::AdvancedTimer::StepVar vtimer("KeepForces");
    const VecConstraintBlockInfo& constraintBlockInfo = hierarchy_constraintBlockInfo[0];
    const VecPersistentID& constraintIds = hierarchy_constraintIds[0];
    // store current force
    _previousForces.resize(_numConstraints);
    for (unsigned int c=0; c<_numConstraints; ++c)
        _previousForces[c] = (*_result)[c];
    // clear previous history
    for (std::map<core::behavior::BaseConstraint*, ConstraintBlockBuf>::iterator it = _previousConstraints.begin(), itend = _previousConstraints.end(); it != itend; ++it)
    {
        ConstraintBlockBuf& buf = it->second;
        for (std::map<PersistentID,int>::iterator it2 = buf.persistentToConstraintIdMap.begin(), it2end = buf.persistentToConstraintIdMap.end(); it2 != it2end; ++it2)
            it2->second = -1;
    }
    // fill info from current ids
    for (unsigned cb = 0; cb < constraintBlockInfo.size(); ++cb)
    {
        const ConstraintBlockInfo& info = constraintBlockInfo[cb];
        if (!info.parent) continue;
        if (!info.hasId) continue;
        ConstraintBlockBuf& buf = _previousConstraints[info.parent];
        int c0 = info.const0;
        int nbl = info.nbLines;
        buf.nbLines = nbl;
        for (int c = 0; c < info.nbGroups; ++c)
            buf.persistentToConstraintIdMap[constraintIds[info.offsetId + c]] = c0 + c*nbl;
    }
}


int LCPConstraintSolver::nlcp_gaussseidel_unbuilt(double *dfree, double *f, std::vector<double>* residuals)
{
    if(!_numConstraints)
        return 0;

    //helper::system::thread::CTime timer;
    double time = 0.0;
    double timeScale = 1000.0 / (double)sofa::helper::system::thread::CTime::getTicksPerSec();
    if ( displayTime.getValue() )
    {
        time = (double) helper::system::thread::CTime::getTime();
    }


    if(_mu==0.0)
    {
        serr<<"WARNING: frictionless case with unbuilt nlcp is not implemented"<<sendl;
        return 0;
    }

    /////// test: numContacts = _numConstraints/3 (must be dividable by 3)
    if (_numConstraints%3 != 0)
    {
        serr<<" WARNING dim should be dividable by 3 in nlcp_gaussseidel"<<sendl;
        return 0;
    }
    int numContacts =  _numConstraints/3;
    //////////////////////////////////////////////
    // iterators
    int it,c1;

    //////////////////////////////////////////////
    // data for iterative procedure
    double _tol = tol.getValue();
    int _maxIt = maxIt.getValue();
    double _mu = mu.getValue();

    //debug
    //std::cout<<"data are set"<<std::endl;


    /// each constraintCorrection has an internal force vector that is set to "0"

    // indirection of the sequence of contact
    std::list<unsigned int> contact_sequence;

    for (unsigned int c=0; c< _numConstraints; c++)
    {
        contact_sequence.push_back(c);
    }


    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetForUnbuiltResolution(f, contact_sequence);

        if(notMuted())
        {
            core::ConstraintParams cparams;
            cc->addComplianceInConstraintSpace(&cparams, _W);
        }
    }




    // return 1;
    if ( displayTime.getValue() )
    {
        dmsg_info() << " build_constraints " << ( (double) timer.getTime() - time)*timeScale<<" ms" ;
        time = (double) timer.getTime();
    }



    //////// Important component if the LCP is not build :
    // for each contact, the pair of constraint correction that is involved with the contact is memorized
    _cclist_elem1.clear();
    _cclist_elem2.clear();
    for (c1=0; c1<numContacts; c1++)
    {
        bool elem1 = false;
        bool elem2 = false;
        for (unsigned int i=0; i<constraintCorrections.size(); i++)
        {

            core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
            if(cc->hasConstraintNumber(3*c1))
            {
                if(elem1)
                {
                    _cclist_elem2.push_back(cc);
                    elem2=true;
                }
                else
                {
                    _cclist_elem1.push_back(cc);
                    elem1=true;
                }

            }
        }
        if(!elem1)
            serr<<"WARNING: no constraintCorrection found for contact"<<c1<<sendl;
        if(!elem2)
            _cclist_elem2.push_back(NULL);

    }

    //debug
    //std::cout<<"_cclist_elem1 _cclist_elem2 are set"<<std::endl;



    // memory allocation of vector d
    unbuilt_d.resize(_numConstraints);
    double *d = &(unbuilt_d[0]);
    //d = (double*)malloc(_numConstraints*sizeof(double));


    // debug
    // std::cout<<"getBlockDiagonalCompliance  Wdiag = "<<(* _Wdiag)<<std::endl;
    // return 1;
    if ( displayTime.getValue() )
    {
        sout<<" link_constraints " << ( (double) timer.getTime() - time)*timeScale<<" ms" <<sendl;
        time = (double) timer.getTime();
    }


    //////////////
    // Beginning of iterative computations
    //////////////


    /////////// the 3x3 diagonal block matrix is built:
    /////////// for each contact, the pair of constraintcorrection is called to add the contribution
    for (c1=0; c1<numContacts; c1++)
    {
        dmsg_info() <<"contact "<<c1<<" cclist_elem1 : "<<_cclist_elem1[c1]->getName();

        // compliance of object1
        _cclist_elem1[c1]->getBlockDiagonalCompliance(_Wdiag, 3*c1, 3*c1+2);

        // compliance of object2 (if object2 exists)
        if(_cclist_elem2[c1] != NULL)
        {
            _cclist_elem2[c1]->getBlockDiagonalCompliance(_Wdiag, 3*c1, 3*c1+2);


           dmsg_info() <<"  _cclist_elem2 : "<<_cclist_elem2[c1]->getName();
        }
        dmsg_info() <<" "<<msgendl;
    }



    // allocation of the inverted system 3x3
    // TODO: evaluate the cost of this step : it can be avoied by directly feeding W33 in constraint correction
    unbuilt_W33.clear();
    unbuilt_W33.resize(numContacts);
    helper::LocalBlock33 *W33 = &(unbuilt_W33[0]); //new helper::LocalBlock33[numContacts];
    //3 = (helper::LocalBlock33 **) malloc (_numConstraints*sizeof(helper::LocalBlock33));
    for (c1=0; c1<numContacts; c1++)
    {
        //3[c1] = new helper::LocalBlock33();
        double w[6];
        w[0] = _Wdiag->element(3*c1  , 3*c1  );
        w[1] = _Wdiag->element(3*c1  , 3*c1+1);
        w[2] = _Wdiag->element(3*c1  , 3*c1+2);
        w[3] = _Wdiag->element(3*c1+1, 3*c1+1);
        w[4] = _Wdiag->element(3*c1+1, 3*c1+2);
        w[5] = _Wdiag->element(3*c1+2, 3*c1+2);
        W33[c1].compute(w[0], w[1] , w[2], w[3], w[4] , w[5]);
    }

    // debug
    dmsg_info() <<" Compliance In constraint Space : \n W ="<<(* _W)<<msgendl
                <<"getBlockDiagonalCompliance   \n Wdiag = "<<(* _Wdiag) ;

    // return 1;
    if (displayTime.getValue())
    {
        dmsg_info() <<" build_diagonal " << ( (double) timer.getTime() - time)*timeScale<<" ms" ;
        time = (double) timer.getTime();
    }

    double error = 0;
    double dn, dt, ds, fn, ft, fs, fn0;

    for (it=0; it<_maxIt; it++)
    {
        std::list<unsigned int>::iterator it_c ;
        error =0;

        for (it_c = contact_sequence.begin(); it_c != contact_sequence.end() ; ++it_c )
        {
            int constraint = *it_c;
            c1 = constraint/3;

            //constraints are treated 3x3 (friction contact)
            ++it_c;
            if(it_c != contact_sequence.end())
                ++it_c;

            // compute the current violation :

            // violation when no contact force
            d[3*c1]=dfree[3*c1]; d[3*c1+1]=dfree[3*c1+1]; d[3*c1+2]=dfree[3*c1+2];


            // set current force in fn, ft, fs
            fn0=fn=f[3*c1]; ft=f[3*c1+1]; fs=f[3*c1+2];

            // displacement of object1 due to contact force
            _cclist_elem1[c1]->addConstraintDisplacement(d, 3*c1, 3*c1+2);

            // displacement of object2 due to contact force (if object2 exists)
            if(_cclist_elem2[c1] != NULL)
                _cclist_elem2[c1]->addConstraintDisplacement(d, 3*c1, 3*c1+2);


            // set displacement in dn, dt, ds
            dn=d[3*c1]; dt=d[3*c1+1]; ds=d[3*c1+2];

            // compute a new state for stick/slip
            /// ATTENTION  NOUVEAU GS_STATE : maintenant dn, dt et ds inclue les forces fn, ft, fs
            W33[c1].New_GS_State(_mu,dn,dt,ds,fn,ft,fs);

            // evaluate an error (based on displacement)
            error += helper::absError(dn,dt,ds,d[3*c1],d[3*c1+1],d[3*c1+2]);

            bool update;
            if (fn0 == 0.0 && fn == 0.0)
                update=false;               // the contact is not active and was not active in the previous step
            else
                update=true;

            // set the new force :
            // compute the Delta of contact forces:
            f[3*c1  ] = fn - f[3*c1  ];
            f[3*c1+1] = ft - f[3*c1+1];
            f[3*c1+2] = fs - f[3*c1+2];

            //std::cout<<"fn = "<< fn<<" -  ft = "<< ft<<" -  fs = "<<fs<<std::endl;
            ///////// verifier si Delta force vaut 0 => pas la peine d'ajouter la force

            // set Delta force on object 1 for evaluating the followings displacement

            if(update)
            {
                _cclist_elem1[c1]->setConstraintDForce(f, 3*c1, 3*c1+2, update);

                // set Delta force on object2 (if object2 exists)
                if(_cclist_elem2[c1] != NULL)
                    _cclist_elem2[c1]->setConstraintDForce(f, 3*c1, 3*c1+2, update);
            }


            ///// debug : verifie si on retrouve le meme dn
            /*
            d[3*c1]=dfree[3*c1]; d[3*c1+1]=dfree[3*c1+1]; d[3*c1+2]=dfree[3*c1+2];
            _cclist_elem1[c1]->addConstraintDisplacement(d, 3*c1, 3*c1+2);
            if(fabs(dn-d[3*c1]) > 0.000000001*fabs(dn) && dn> 0.1*_tol)
            msg_info()<<"WARNING debug : dn ="<<dn<<" d["<<3*c1<<"]= "<< d[3*c1]<<" dfree= "<<dfree[3*c1]<<"  - update :"<<update<<" with fn ="<<fn<<" and f["<<3*c1<<"]= "<< fn-f[3*c1  ]<<std::endl;
            */

            // set force on the contact force vector
            helper::set3Dof(f,c1,fn,ft,fs);
        }

        residuals->push_back(error);


        if (error < _tol*(numContacts+1))
        {
            msg_info_when(displayTime.getValue()) << "convergence after "<<it<<" iterations - error"<<error<<msgendl
                                                  <<" GAUSS_SEIDEL iterations  " << ( (double) timer.getTime() - time)*timeScale<<" ms";


            sofa::helper::AdvancedTimer::valSet("GS iterations", it+1);
            return 1;
        }

    }

    sofa::helper::AdvancedTimer::valSet("GS iterations", it);

    msg_info_when( displayTime.getValue() ) <<" GAUSS_SEIDEL iterations  "
                                           << ( (double) timer.getTime() - time)*timeScale<<" ms" ;

    msg_error() << "No convergence in  unbuilt nlcp gaussseidel function : error ="
                <<error <<" after"<< it<<" iterations";

    return 0;
}





int LCPConstraintSolver::lcp_gaussseidel_unbuilt(double *dfree, double *f, std::vector<double>* /*residuals*/)
{
    //helper::system::thread::CTime timer;
    double time = 0.0;
    double timeScale = 1.0;
    if ( displayTime.getValue() )
    {
        time = (double) helper::system::thread::CTime::getTime();
        timeScale = 1000.0 / (double)sofa::helper::system::thread::CTime::getTicksPerSec();
    }


    if(_mu!=0.0)
    {
        dmsg_warning() <<"friction case with unbuilt lcp is not implemented" ;
        return 0;
    }

    int numContacts =  _numConstraints;
    //////////////////////////////////////////////
    // iterators
    int it,c1;

    //////////////////////////////////////////////
    // data for iterative procedure
    double _tol = tol.getValue();
    int _maxIt = maxIt.getValue();

    // indirection of the sequence of contact
    std::list<unsigned int> contact_sequence;

    for (unsigned int c=0; c< _numConstraints; c++)
    {
        contact_sequence.push_back(c);
    }


    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetForUnbuiltResolution(f, contact_sequence);
    }

    if ( displayTime.getValue() )
    {
        msg_info() << " build_constraints " << ( (double) timer.getTime() - time)*timeScale<<" ms" ;
        time = (double) timer.getTime();
    }



    //////// Important component if the LCP is not build :
    // for each contact, the pair of constraint correction that is involved with the contact is memorized
    _cclist_elem1.resize(numContacts);
    _cclist_elem2.resize(numContacts);
    for (c1=0; c1<numContacts; c1++)
    {
        bool elem1 = false;
        bool elem2 = false;
        for (unsigned int i=0; i<constraintCorrections.size(); i++)
        {

            core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
            if(cc->hasConstraintNumber(c1))
            {
                if(elem1)
                {
                    _cclist_elem2[c1] = (cc);
                    elem2=true;
                }
                else
                {
                    _cclist_elem1[c1] = (cc);
                    elem1=true;
                }

            }
        }
        msg_warning_when(!elem1) << "WARNING: no constraintCorrection found for contact"<<c1 ;
        if(!elem2)
            _cclist_elem2[c1] = (NULL);
    }

    unbuilt_d.resize(_numConstraints);
    double *d = &(unbuilt_d[0]);

    if ( displayTime.getValue() )
    {
        msg_info() <<" link_constraints " << ( (double) timer.getTime() - time)*timeScale<<" ms" ;
        time = (double) timer.getTime();
    }

    //////////////
    // Beginning of iterative computations
    //////////////

    // the 1x1 diagonal block matrix is built:
    // for each contact, the pair of constraintcorrection is called to add the contribution
    for (c1=0; c1<numContacts; c1++)
    {
        // compliance of object1
        _cclist_elem1[c1]->getBlockDiagonalCompliance(_Wdiag, c1, c1);
        // compliance of object2 (if object2 exists)
        if(_cclist_elem2[c1] != NULL)
        {
            _cclist_elem2[c1]->getBlockDiagonalCompliance(_Wdiag, c1, c1);
        }
    }
    // std::cout<<"getBlockDiagonalCompliance  Wdiag = "<<(* _Wdiag)<<std::endl;

    unbuilt_W11.resize(numContacts);
    //unbuilt_invW11.resize(numContacts);
    double *W11 = &(unbuilt_W11[0]);
    //double *invW11 = &(unbuilt_invW11[0]);
    for (c1=0; c1<numContacts; c1++)
    {
        W11[c1] = _Wdiag->element(c1, c1);
        //invW11[c1] = 1.0 / W11[c1];
    }

    if ( displayTime.getValue() )
    {
        msg_info() <<" build_diagonal " << ( (double) timer.getTime() - time)*timeScale<<" ms" ;
        time = (double) timer.getTime();
    }

    double error = 0;
    double dn, fn, fn0;

    for (it=0; it<_maxIt; it++)
    {
        std::list<unsigned int>::iterator it_c;
        error =0;

        for (it_c = contact_sequence.begin(); it_c != contact_sequence.end(); ++it_c)
        {

            c1 = *it_c;

            // compute the current violation :
            // violation when no contact force
            d[c1]=dfree[c1];
            // set current force in fn
            fn0=fn=f[c1];

            // displacement of object1 due to contact force
            _cclist_elem1[c1]->addConstraintDisplacement(d, c1, c1);
            // displacement of object2 due to contact force (if object2 exists)
            if(_cclist_elem2[c1] != NULL)
                _cclist_elem2[c1]->addConstraintDisplacement(d, c1, c1);
            // set displacement in dn
            dn=d[c1];

            // compute a new state for stick/slip
            /// ATTENTION  NOUVEAU GS_STATE : maintenant dn inclue les forces fn
            //W33[c1].New_GS_State(_mu,dn,dt,ds,fn,ft,fs);
            fn -= dn / W11[c1];
            if (fn < 0) fn = 0;
            error += fabs(W11[c1] * (fn - fn0));

            bool update = (fn0 != 0.0 || fn != 0.0);

            if(update)
            {
                // set the new force :
                // compute the Delta of contact forces:
                f[c1] = fn - fn0;
                _cclist_elem1[c1]->setConstraintDForce(f, c1, c1, update);
                if(_cclist_elem2[c1] != NULL)
                    _cclist_elem2[c1]->setConstraintDForce(f, c1, c1, update);
            }

            f[c1] = fn;
        }

        if (error < _tol*(numContacts+1))
        {
            if ( displayTime.getValue() )
            {
                msg_info() <<"convergence after "<<it<<" iterations - error = "<<error << msgendl
                           <<" GAUSS_SEIDEL iterations  " << ( (double) timer.getTime() - time)*timeScale<<" ms" ;
            }
            sofa::helper::AdvancedTimer::valSet("GS iterations", it+1);

            return 1;
        }
    }

    msg_info_when(displayTime.getValue()) <<" GAUSS_SEIDEL iterations "
                                         << ( (double) timer.getTime() - time)*timeScale<<" ms" ;


    sofa::helper::AdvancedTimer::valSet("GS iterations", it);

    msg_error() <<" No convergence in  unbuilt lcp gaussseidel function : error ="
                <<error <<" after"<< it<<" iterations";

    return 0;
}

ConstraintProblem* LCPConstraintSolver::getConstraintProblem()
{
    return last_lcp;
}

void LCPConstraintSolver::lockConstraintProblem(ConstraintProblem* l1, ConstraintProblem* l2)
{
    if((lcp!=l1)&&(lcp!=l2)) // Le lcp courant n'est pas locké
        return;

    if((&lcp1!=l1)&&(&lcp1!=l2)) // lcp1 n'est pas locké
        lcp = &lcp1;
    else if((&lcp2!=l1)&&(&lcp2!=l2)) // lcp2 n'est pas locké
        lcp = &lcp2;
    else
        lcp = &lcp3; // lcp1 et lcp2 sont lockés, donc lcp3 n'est pas locké

    // Mise �  jour de _W _dFree et _result
    _W = &lcp->W;
    _dFree = &lcp->dFree;
    _result = &lcp->f;
}


void LCPConstraintSolver::draw(const core::visual::VisualParams* vparams)
{
    unsigned int showLevels = (unsigned int) this->showLevels.getValue();
    if (showLevels > hierarchy_constraintBlockInfo.size()) showLevels = hierarchy_constraintBlockInfo.size();
    if (!showLevels) return;
    double showCellWidth = this->showCellWidth.getValue();
    defaulttype::Vector3 showTranslation = this->showTranslation.getValue();
    defaulttype::Vector3 showLevelTranslation = this->showLevelTranslation.getValue();

    const int merge_spatial_step = this->merge_spatial_step.getValue();
    const int merge_spatial_shift = 0; // merge_spatial_step/2
    const int merge_local_levels = this->merge_local_levels.getValue();

    // from http://colorexplorer.com/colormatch.aspx
    const unsigned int colors[72]= { 0x2F2FBA, 0x111145, 0x2FBA8C, 0x114534, 0xBA8C2F, 0x453411, 0x2F72BA, 0x112A45, 0x2FBA48, 0x11451B, 0xBA2F5B, 0x451122, 0x2FB1BA, 0x114145, 0x79BA2F, 0x2D4511, 0x9E2FBA, 0x3B1145, 0x2FBA79, 0x11452D, 0xBA662F, 0x452611, 0x2F41BA, 0x111845, 0x2FBA2F, 0x114511, 0xBA2F8C, 0x451134, 0x2F8CBA, 0x113445, 0x6DBA2F, 0x284511, 0xAA2FBA, 0x3F1145, 0x2FAABA, 0x113F45, 0xAFBA2F, 0x414511, 0x692FBA, 0x271145, 0x2FBAAA, 0x11453F, 0xBA892F, 0x453311, 0x2F31BA, 0x111245, 0x2FBA89, 0x114533, 0xBA4F2F, 0x451D11, 0x2F4DBA, 0x111C45, 0x2FBA6D, 0x114528, 0xBA2F56, 0x451120, 0x2F72BA, 0x112A45, 0x2FBA48, 0x11451B, 0xBA2F9A, 0x451139, 0x2F93BA, 0x113645, 0x3FBA2F, 0x174511, 0x662FBA, 0x261145, 0x2FBAA8, 0x11453E, 0xB1BA2F, 0x414511};

    union
    {
        int i;
        unsigned char b[4];
    } color;

    int coord0 = 0;
    int coordFact = 1;
    for (unsigned int level = 0; level < showLevels; ++level)
    {
        const VecConstraintBlockInfo& constraintBlockInfo = hierarchy_constraintBlockInfo[level];
        const VecConstCoord&          constraintPositions = hierarchy_constraintPositions[level];
        const VecConstDeriv&          constraintDirections = hierarchy_constraintDirections[level];
        const VecConstArea&           constraintAreas = hierarchy_constraintAreas[level];

        for (unsigned cb = 0; cb < constraintBlockInfo.size(); ++cb)
        {
            const ConstraintBlockInfo& info = constraintBlockInfo[cb];
            if (!info.hasPosition)
                continue;
            if (!info.hasDirection)
                continue;

            const int c0 = info.const0;
            const int nbl = info.nbLines;
            for (int c = 0; c < info.nbGroups; ++c)
            {
                int idFine = c0 + c*nbl;
                if ((unsigned)(info.offsetPosition + c) >= constraintPositions.size())
                {
                    msg_info() << "Level " << level << ": constraint " << idFine << " from " << (info.parent ? info.parent->getName() : std::string("NULL")) << " has invalid position index" ;
                    break;
                }
                if ((unsigned)(info.offsetDirection + 3*c) >= constraintDirections.size())
                {
                    msg_info() << "Level " << level << ": constraint " << idFine << " from " << (info.parent ? info.parent->getName() : std::string("NULL")) << " has invalid direction index" ;
                    break;
                }
                ConstCoord posFine = constraintPositions[info.offsetPosition + c];
                ConstDeriv dirFineN  = constraintDirections[info.offsetDirection + 3*c + 0];
                ConstDeriv dirFineT1 = constraintDirections[info.offsetDirection + 3*c + 1];
                ConstDeriv dirFineT2 = constraintDirections[info.offsetDirection + 3*c + 2];
                ConstArea area = (info.hasArea) ? constraintAreas[info.offsetArea + c] : (ConstArea)(2*coordFact*coordFact*showCellWidth*showCellWidth);

                defaulttype::Vector3 centerFine = showTranslation + showLevelTranslation*level;
                for (int i=0; i<3; ++i) centerFine[i] += ((posFine[i]+0.5)*coordFact + coord0) * showCellWidth;
                double radius = sqrt(area*0.5);

                int colid = (level * 12 + ((int)level < merge_local_levels ? (cb % 2) : 0)) % 72;
                color.i = colors[colid + 0];
                vparams->drawTool()->drawArrow(
                    centerFine,centerFine+dirFineN*radius*2.0f,
                    (float)radius*2.0f*0.03f,
                    defaulttype::Vec<4,float>((float)(color.b[0]) * (1.0f/255.0f),
                            (float)(color.b[1]) * (1.0f/255.0f),
                            (float)(color.b[2]) * (1.0f/255.0f),
                            1.0f));
                if (_mu > 1.0e-6)
                {
                    color.i = colors[colid + 2];
                    vparams->drawTool()->drawArrow(
                        centerFine-dirFineT1*radius*_mu,centerFine+dirFineT1*radius*_mu,
                        (float)(radius*_mu*0.03f),
                        defaulttype::Vec<4,float>((float)(color.b[0]) * (1.0f/255.0f),
                                (float)(color.b[1]) * (1.0f/255.0f),
                                (float)(color.b[2]) * (1.0f/255.0f),
                                1.0f));
                    color.i = colors[colid + 4];
                    vparams->drawTool()->drawArrow(
                        centerFine-dirFineT2*radius*_mu,centerFine+dirFineT2*radius*_mu,
                        (float)(radius*_mu*0.03f),
                        defaulttype::Vec<4,float>(color.b[0] * (1.0f/255.0f),
                                color.b[1] * (1.0f/255.0f),
                                color.b[2] * (1.0f/255.0f),
                                1.0f));
                }
            }
        }
        coord0 = (coord0 - merge_spatial_shift) * merge_spatial_step;
        coordFact *= merge_spatial_step;
    }
}

int LCPConstraintSolverClass = core::RegisterObject("A Constraint Solver using the Linear Complementarity Problem formulation to solve BaseConstraint based components")
        .add< LCPConstraintSolver >();

SOFA_DECL_CLASS(LCPConstraintSolver);


} // namespace constraintset

} // namespace component

} // namespace sofa
