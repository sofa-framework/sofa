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

#include <sofa/component/constraint/lagrangian/solver/LCPConstraintSolver.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>

#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/fwd.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalGetConstraintInfoVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalGetConstraintInfoVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalVOpVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalVOpVisitor;

using sofa::core::VecId;

namespace sofa::component::constraint::lagrangian::solver
{

LCPConstraintSolver::LCPConstraintSolver()
    : displayDebug(initData(&displayDebug, false, "displayDebug","Display debug information."))
    , initial_guess(initData(&initial_guess, true, "initial_guess","activate LCP results history to improve its resolution performances."))
    , build_lcp(initData(&build_lcp, true, "build_lcp", "LCP is not fully built to increase performance in some case."))
    , tol( initData(&tol, 0.001_sreal, "tolerance", "residual error threshold for termination of the Gauss-Seidel algorithm"))
    , maxIt( initData(&maxIt, 1000, "maxIt", "maximal number of iterations of the Gauss-Seidel algorithm"))
    , mu( initData(&mu, 0.6_sreal, "mu", "Friction coefficient"))
    , minW( initData(&minW, 0.0_sreal, "minW", "If not zero, constraints whose self-compliance (i.e. the corresponding value on the diagonal of W) is smaller than this threshold will be ignored"))
    , maxF( initData(&maxF, 0.0_sreal, "maxF", "If not zero, constraints whose response force becomes larger than this threshold will be ignored"))
    , multi_grid(initData(&multi_grid, false, "multi_grid","activate multi_grid resolution (NOT STABLE YET)"))
    , multi_grid_levels(initData(&multi_grid_levels, 2, "multi_grid_levels","if multi_grid is active: how many levels to create (>=2)"))
    , merge_method( initData(&merge_method, 0, "merge_method","if multi_grid is active: which method to use to merge constraints (0 = compliance-based, 1 = spatial coordinates)"))
    , merge_spatial_step( initData(&merge_spatial_step, 2, "merge_spatial_step", "if merge_method is 1: grid size reduction between multigrid levels"))
    , merge_local_levels( initData(&merge_local_levels, 2, "merge_local_levels", "if merge_method is 1: up to the specified level of the multigrid, constraints are grouped locally, i.e. separately within each contact pairs, while on upper levels they are grouped globally independently of contact pairs."))
    , d_constraintForces(initData(&d_constraintForces,"constraintForces","OUTPUT: constraint forces (stored only if computeConstraintForces=True)"))
    , d_computeConstraintForces(initData(&d_computeConstraintForces,false,
                                        "computeConstraintForces",
                                        "enable the storage of the constraintForces."))
    , constraintGroups( initData(&constraintGroups, "group", "list of ID of groups of constraints to be handled by this solver."))
    , f_graph( initData(&f_graph,"graph","Graph of residuals at each iteration"))
    , showLevels( initData(&showLevels,0,"showLevels","Number of constraint levels to display"))
    , showCellWidth( initData(&showCellWidth, "showCellWidth", "Distance between each constraint cells"))
    , showTranslation( initData(&showTranslation, "showTranslation", "Position of the first cell"))
    , showLevelTranslation( initData(&showLevelTranslation, "showLevelTranslation", "Translation between levels"))
    , current_cp(&lcp1)
    , last_cp(nullptr)
    , _W(&lcp1.W)
    , _dFree(&lcp1.dFree)
    , _result(&lcp1.f)
{
    _numConstraints = 0;
    constraintGroups.beginEdit()->insert(0);
    constraintGroups.endEdit();

    f_graph.setWidget("graph");

    tol.setRequired(true);
    maxIt.setRequired(true);
}

LCPConstraintSolver::~LCPConstraintSolver()
{
}

void LCPConstraintProblem::solveTimed(SReal tolerance, int maxIt, SReal timeout)
{
    helper::nlcp_gaussseidelTimed(dimension, getDfree(), getW(), getF(), mu, tolerance, maxIt, true, timeout);
}

bool LCPConstraintSolver::prepareStates(const core::ConstraintParams * /*cParams*/, MultiVecId /*res1*/, MultiVecId /*res2*/)
{
    last_cp = current_cp;
    MechanicalVOpVisitor(core::execparams::defaultInstance(), (core::VecId)core::VecDerivId::dx()).setMapped(true).execute( getContext()); //dX=0

    msg_info() <<" propagate DXn performed - collision called" ;

    SCOPED_TIMER("resetContactForce");
  
    for (const auto& cc : l_constraintCorrections)
    {
        cc->resetContactForce();
    }

    return true;
}

bool LCPConstraintSolver::buildSystem(const core::ConstraintParams * /*cParams*/, MultiVecId res1, MultiVecId res2)
{
    SOFA_UNUSED(res1);
    SOFA_UNUSED(res2);

    buildSystem();
    return true;
}


void LCPConstraintSolver::buildSystem()
{
    core::ConstraintParams cparams;

    cparams.setX(core::ConstVecCoordId::freePosition());
    cparams.setV(core::ConstVecDerivId::freeVelocity());

    _numConstraints = buildConstraintMatrix(&cparams);
    sofa::helper::AdvancedTimer::valSet("numConstraints", _numConstraints);

    current_cp->mu = mu.getValue();
    current_cp->clear(_numConstraints);

    getConstraintViolation(&cparams, _dFree);

    if (build_lcp.getValue())
    {
        addComplianceInConstraintSpace(cparams);
    }
    else
    {
        // When the LCP or the NLCP is not fully built, the  diagonal blocks of the matrix are still needed for the resolution
        _Wdiag.resize(_numConstraints,_numConstraints);
    }

    buildHierarchy();

    getConstraintInfo(cparams);

}

bool LCPConstraintSolver::solveSystem(const core::ConstraintParams * /*cParams*/, MultiVecId /*res1*/, MultiVecId /*res2*/)
{
    const auto _mu = mu.getValue();

    std::map < std::string, sofa::type::vector<SReal> >& graph = *f_graph.beginEdit();

    if (build_lcp.getValue())
    {
        const SReal _tol = tol.getValue();
        const int _maxIt = maxIt.getValue();
        const SReal _minW = minW.getValue();
        const SReal _maxF = maxF.getValue();

        if (_mu > 0.0)
        {
            current_cp->tolerance = _tol;

            if (multi_grid.getValue())
            {
                {
                    SCOPED_TIMER("ConstraintsMerge");
                    MultigridConstraintsMerge();
                }

                sofa::type::vector<SReal>& graph_residuals = graph["Error"];
                graph_residuals.clear();
                sofa::type::vector<SReal>& graph_violations = graph["Violation"];
                graph_violations.clear();
                sofa::type::vector<SReal>& graph_levels = graph["Level"];
                graph_levels.clear();

                {
                    SCOPED_TIMER("NLCP MultiGrid");
                    helper::nlcp_multiGrid_Nlevels(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _mu, _tol, _maxIt, initial_guess.getValue(),
                           hierarchy_contact_group, hierarchy_num_group, hierarchy_constraint_group, hierarchy_constraint_group_fact,  notMuted(), &graph_residuals, &graph_levels, &graph_violations);
                }

            }
            else
            {
                sofa::type::vector<SReal>& graph_error = graph["Error"];
                graph_error.clear();
                sofa::type::vector<SReal>& graph_violations = graph["Violation"];
                graph_violations.clear();

                {
                    SCOPED_TIMER("NLCP GaussSeidel");
                    helper::nlcp_gaussseidel(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _mu, _tol, _maxIt, initial_guess.getValue(),
                           notMuted(), _minW, _maxF, &graph_error, &graph_violations);
                }
             }
        }
        else
        {
            sofa::type::vector<SReal>& graph_error = graph["Error"];
            graph_error.clear();

            {
                SCOPED_TIMER("LCP GaussSeidel");
                helper::gaussSeidelLCP1(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _tol, _maxIt, _minW, _maxF, &graph_error);
            }
            if (notMuted())
            {
                helper::printLCP(_dFree->ptr(), _W->lptr(), _result->ptr(),_numConstraints);
            }
        }
    }
    else
    {

        sofa::type::vector<SReal>& graph_error = graph["Error"];
        graph_error.clear();

        {
            SCOPED_TIMER("NLCP GaussSeidel Unbuild");
            gaussseidel_unbuilt(_dFree->ptr(), _result->ptr(), &graph_error);
        }

        if (displayDebug.getValue())
        {
            dmsg_info() <<"_result unbuilt:"<<(*_result) ;

            _result->resize(_numConstraints);

            const SReal _tol = tol.getValue();
            const int _maxIt = maxIt.getValue();

            buildSystem();

            helper::nlcp_gaussseidel(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _mu, _tol, _maxIt, initial_guess.getValue());
            dmsg_info() <<"\n_result nlcp :"<<(*_result);
        }
    }

    f_graph.endEdit();

    if(d_computeConstraintForces.getValue())
    {
        sofa::helper::WriteOnlyAccessor<Data<type::vector<SReal>>> constraints = d_constraintForces;
        constraints.resize(current_cp->getDimension());
        for(int i=0; i<current_cp->getDimension(); i++)
        {
            constraints[i] = _result->ptr()[i];
        }
    }

    return true;
}

bool LCPConstraintSolver::applyCorrection(const core::ConstraintParams * /*cParams*/, MultiVecId /*res1*/, MultiVecId /*res2*/)
{
    if (initial_guess.getValue())
    {
        keepContactForcesValue();
    }

    SCOPED_TIMER("Apply Contact Force");
    for (const auto& l_constraintCorrection : l_constraintCorrections)
    {
        if (!l_constraintCorrection->getContext()->isSleeping())
        {
            l_constraintCorrection->applyContactForce(_result);
        }
    }

    return true;
}

void LCPConstraintSolver::buildHierarchy()
{
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
}

void LCPConstraintSolver::getConstraintInfo(core::ConstraintParams cparams)
{
    if ((initial_guess.getValue() || multi_grid.getValue() || showLevels.getValue()) && (_numConstraints != 0))
    {
        {
            SCOPED_TIMER("Get Constraint Info");
            MechanicalGetConstraintInfoVisitor(&cparams, hierarchy_constraintBlockInfo[0], hierarchy_constraintIds[0], hierarchy_constraintPositions[0], hierarchy_constraintDirections[0], hierarchy_constraintAreas[0]).execute(getContext());
        }
        if (initial_guess.getValue())
        {
            computeInitialGuess();
        }
    }
}

void LCPConstraintSolver::addComplianceInConstraintSpace(core::ConstraintParams cparams)
{
    SCOPED_TIMER("Get Compliance");

    dmsg_info() <<" computeCompliance in "  << l_constraintCorrections.size() << " constraintCorrections" ;

    for (const auto& cc : l_constraintCorrections)
    {
        cc->addComplianceInConstraintSpace(&cparams, _W);
    }

    dmsg_info() << "W=" << *_W ;
}

void LCPConstraintSolver::build_Coarse_Compliance(std::vector<int> &constraint_merge, int sizeCoarseSystem)
{
    /* constraint_merge => tableau donne l'indice du groupe de contraintes dans le système grossier en fonction de l'indice de la contrainte dans le système de départ */
    dmsg_info() <<"build_Coarse_Compliance is called : size="<<sizeCoarseSystem ;

    _Wcoarse.clear();

    dmsg_error_when(sizeCoarseSystem==0) <<"no constraint" ;

    _Wcoarse.resize(sizeCoarseSystem,sizeCoarseSystem);
    for (unsigned int i=0; i<l_constraintCorrections.size(); i++)
    {
        core::behavior::BaseConstraintCorrection* cc = l_constraintCorrections[i];
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
        msg_error() << "Unsupported merge method " << merge_method.getValue();
    }
}

void LCPConstraintSolver::MultigridConstraintsMerge_Compliance()
{
    const SReal criterion=0.0;
    const int numContacts = _numConstraints/3;

    hierarchy_contact_group.resize(1);
    hierarchy_constraint_group.resize(1);
    hierarchy_constraint_group_fact.resize(1);
    hierarchy_num_group.resize(1);
    std::vector<int> group_lead;
    std::vector<int>& contact_group = hierarchy_contact_group[0];
    std::vector<int>& constraint_group = hierarchy_constraint_group[0];
    std::vector<SReal>& constraint_group_fact = hierarchy_constraint_group_fact[0];
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
    constexpr int merge_spatial_shift = 0; // merge_spatial_step/2
    const int merge_local_levels = this->merge_local_levels.getValue();
    int numConstraints = _numConstraints;
    int numContacts = numConstraints/3;
    int nLevels = multi_grid_levels.getValue();
    if (nLevels < 2) nLevels = 2;

    msg_info() << "Multigrid merge from " << numContacts << " contacts.";

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
        std::vector<SReal>& constraint_group_fact = hierarchy_constraint_group_fact[level-1];
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
            msg_info() << "MultigridConstraintsMerge_Spatial level " << level-1 << " constraint block " << cb << " from " << (info.parent ? info.parent->getName() : std::string("nullptr"))
                    << " : c0 = " << info.const0 << " nbl = " << info.nbLines << " nbg = " << info.nbGroups << " offsetPosition = " << info.offsetPosition << " offsetDirection = " << info.offsetDirection << " offsetArea = " << info.offsetArea;
            if (!info.hasPosition)
            {
                msg_error() << "MultigridConstraintsMerge_Spatial: constraints from " << (info.parent ? info.parent->getName() : std::string("nullptr")) << " have no position data";
                continue;
            }
            if (!info.hasDirection)
            {
                msg_error() << "MultigridConstraintsMerge_Spatial: constraints from " << (info.parent ? info.parent->getName() : std::string("nullptr")) << " have no direction data";
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
                    msg_error() << "MultigridConstraintsMerge_Spatial level " << level << ": constraint " << idFine << " from " << (info.parent ? info.parent->getName() : std::string("nullptr")) << " has invalid index";
                    break;
                }
                if ((unsigned)(info.offsetPosition + c) >= constraintPositions.size())
                {
                    msg_error() << "MultigridConstraintsMerge_Spatial level " << level << ": constraint " << idFine << " from " << (info.parent ? info.parent->getName() : std::string("nullptr")) << " has invalid position index";
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
                auto [insertIt, insertSuccess] = coord2coarseId.insert(std::map<ConstCoord,int>::value_type(posCoarse, (int)num_group));
                int idCoarse = insertIt->second * 3;
                if (insertSuccess)
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
                    SReal dotNN   = dirCoarseN  * dirFineN;
                    SReal dotT1T1 = dirCoarseT1 * dirFineT1;
                    SReal dotT2T2 = dirCoarseT2 * dirFineT2;
                    SReal dotT2T1 = dirCoarseT2 * dirFineT1;
                    SReal dotT1T2 = dirCoarseT1 * dirFineT2;
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
        msg_info() << "Multigrid merge level " << level << ": " << num_group << " groups.";

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
            SReal dotNN   = dirCoarseN  * dirFineN;
            if (dotNN < 0)
            {
                // constraint direction is flipped, so relative velocities for friction are reversed
                dirFineT1 = -dirFineT1;
                dirFineT2 = -dirFineT2;
            }

            SReal dotT1T1 = dirCoarseT1 * dirFineT1;
            SReal dotT2T2 = dirCoarseT2 * dirFineT2;
            SReal dotT2T1 = dirCoarseT2 * dirFineT1;
            SReal dotT1T2 = dirCoarseT1 * dirFineT2;
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
        msg_info() << "MultigridConstraintsMerge_Spatial level " << nLevels-1 << " constraint block " << cb << " from " << (info.parent ? info.parent->getName() : std::string("nullptr"))
                << " : c0 = " << info.const0 << " nbl = " << info.nbLines << " nbg = " << info.nbGroups << " offsetPosition = " << info.offsetPosition << " offsetDirection = " << info.offsetDirection << " offsetArea = " << info.offsetArea;
    }
}

void LCPConstraintSolver::computeInitialGuess()
{
    sofa::helper::AdvancedTimer::StepVar vtimer("InitialGuess");

    const auto _mu = mu.getValue();
    const VecConstraintBlockInfo& constraintBlockInfo = hierarchy_constraintBlockInfo[0];
    const VecPersistentID& constraintIds = hierarchy_constraintIds[0];
    const int numContact = (_mu > 0.0) ? _numConstraints/3 : _numConstraints;

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
        }
    }
    for (const ConstraintBlockInfo& info : constraintBlockInfo)
    {
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
            const int prevIndex = it->second;
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
    for (auto& previousConstraint : _previousConstraints)
    {
        ConstraintBlockBuf& buf = previousConstraint.second;
        for (auto& it2 : buf.persistentToConstraintIdMap)
            it2.second = -1;
    }
    // fill info from current ids
    for (const ConstraintBlockInfo& info : constraintBlockInfo)
    {
        if (!info.parent) continue;
        if (!info.hasId) continue;
        ConstraintBlockBuf& buf = _previousConstraints[info.parent];
        const int c0 = info.const0;
        const int nbl = info.nbLines;
        buf.nbLines = nbl;
        for (int c = 0; c < info.nbGroups; ++c)
        {
            buf.persistentToConstraintIdMap[constraintIds[info.offsetId + c]] = c0 + c*nbl;
        }
    }
}


int LCPConstraintSolver::nlcp_gaussseidel_unbuilt(SReal *dfree, SReal *f, std::vector<SReal>* residuals)
{
    if(!_numConstraints)
        return 0;

    auto _mu = mu.getValue();
    if(_mu==0.0)
    {
        msg_error() << "frictionless case with unbuilt nlcp is not implemented";
        return 0;
    }

    if (_numConstraints%3 != 0)
    {
        msg_error() << "dim should be dividable by 3 in nlcp_gaussseidel";
        return 0;
    }

    sofa::helper::advancedtimer::stepBegin("build_constraints");

    int numContacts =  _numConstraints/3;

    int it,c1;

    // data for iterative procedure
    SReal _tol = tol.getValue();
    int _maxIt = maxIt.getValue();

    /// each constraintCorrection has an internal force vector that is set to "0"

    // indirection of the sequence of contact
    std::list<unsigned int> contact_sequence;

    for (unsigned int c=0; c< _numConstraints; c++)
    {
        contact_sequence.push_back(c);
    }


    for (unsigned int i=0; i<l_constraintCorrections.size(); i++)
    {
        core::behavior::BaseConstraintCorrection* cc = l_constraintCorrections[i];
        cc->resetForUnbuiltResolution(f, contact_sequence);
    }

    sofa::helper::advancedtimer::stepEnd("build_constraints");

    auto linkConstraintTimer = std::make_unique<sofa::helper::ScopedAdvancedTimer>("link_constraints");

    //////// Important component if the LCP is not build :
    // for each contact, the pair of constraint correction that is involved with the contact is memorized
    _cclist_elem1.clear();
    _cclist_elem2.clear();
    std::vector<int> missingConstraintCorrectionContacts;
    for (c1=0; c1<numContacts; c1++)
    {
        bool elem1 = false;
        bool elem2 = false;

        for (const auto& cc : l_constraintCorrections)
        {
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
        if (!elem1)
        {
            _cclist_elem1.push_back(nullptr);
            missingConstraintCorrectionContacts.push_back(c1);
        }
        if(!elem2)
            _cclist_elem2.push_back(nullptr);

    }

    if (!missingConstraintCorrectionContacts.empty())
    {
        std::stringstream ss;
        for (const auto c : missingConstraintCorrectionContacts)
        {
            ss << c << ' ';
        }
        msg_error() << "The following contacts do not have an associated constraint correction component: " << ss.str();
        if (missingConstraintCorrectionContacts.size() == _cclist_elem1.size())
        {
            msg_error() << "None of the contacts has an associated constraint correction component: constraint correction is aborted";
            return 0;
        }
    }

    // memory allocation of vector d
    unbuilt_d.resize(_numConstraints);
    SReal *d = &(unbuilt_d[0]);

    linkConstraintTimer.reset();
    auto buildDiagonalTimer = std::make_unique<sofa::helper::ScopedAdvancedTimer>("build_diagonal");

    //////////////
    // Beginning of iterative computations
    //////////////


    /////////// the 3x3 diagonal block matrix is built:
    /////////// for each contact, the pair of constraintcorrection is called to add the contribution
    for (c1=0; c1<numContacts; c1++)
    {
        dmsg_info() << "contact " << c1 << " cclist_elem1 : " << _cclist_elem1[c1]->getName();

        // compliance of object1
        if (_cclist_elem1[c1] != nullptr)
        {
            _cclist_elem1[c1]->getBlockDiagonalCompliance(&_Wdiag, 3 * c1, 3 * c1 + 2);
        }
        // compliance of object2 (if object2 exists)
        if(_cclist_elem2[c1] != nullptr)
        {
            _cclist_elem2[c1]->getBlockDiagonalCompliance(&_Wdiag, 3*c1, 3*c1+2);


           dmsg_info() <<"  _cclist_elem2 : "<<_cclist_elem2[c1]->getName();
        }
        dmsg_info() <<" "<<msgendl;
    }



    // allocation of the inverted system 3x3
    // TODO: evaluate the cost of this step : it can be avoied by directly feeding W33 in constraint correction
    unbuilt_W33.clear();
    unbuilt_W33.resize(numContacts);
    helper::LocalBlock33 *W33 = &(unbuilt_W33[0]); //new helper::LocalBlock33[numContacts];
    for (c1=0; c1<numContacts; c1++)
    {
        SReal w[6];
        w[0] = _Wdiag.element(3*c1  , 3*c1  );
        w[1] = _Wdiag.element(3*c1  , 3*c1+1);
        w[2] = _Wdiag.element(3*c1  , 3*c1+2);
        w[3] = _Wdiag.element(3*c1+1, 3*c1+1);
        w[4] = _Wdiag.element(3*c1+1, 3*c1+2);
        w[5] = _Wdiag.element(3*c1+2, 3*c1+2);
        W33[c1].compute(w[0], w[1] , w[2], w[3], w[4] , w[5]);
    }

    dmsg_info() <<" Compliance In constraint Space : \n W ="<<(* _W)<<msgendl
                <<"getBlockDiagonalCompliance   \n Wdiag = "<< _Wdiag ;

    buildDiagonalTimer.reset();

    SCOPED_TIMER_VARNAME(gaussSeidelTimer, "GAUSS_SEIDEL");

    SReal error = 0;
    SReal dn, dt, ds, fn, ft, fs, fn0;

    for (it=0; it<_maxIt; it++)
    {
        std::list<unsigned int>::iterator it_c ;
        error =0;

        //constraints are treated 3x3 (friction contact)
        for (it_c = contact_sequence.begin(); it_c != contact_sequence.end() ; std::advance(it_c, 3) )
        {
            int constraint = *it_c;
            c1 = constraint/3;

            // compute the current violation :

            // violation when no contact force
            d[3*c1]=dfree[3*c1]; d[3*c1+1]=dfree[3*c1+1]; d[3*c1+2]=dfree[3*c1+2];


            // set current force in fn, ft, fs
            fn0=fn=f[3*c1]; ft=f[3*c1+1]; fs=f[3*c1+2];

            // displacement of object1 due to contact force
            if(_cclist_elem1[c1] != nullptr)
                _cclist_elem1[c1]->addConstraintDisplacement(d, 3*c1, 3*c1+2);

            // displacement of object2 due to contact force (if object2 exists)
            if(_cclist_elem2[c1] != nullptr)
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

            ///////// verifier si Delta force vaut 0 => pas la peine d'ajouter la force

            // set Delta force on object 1 for evaluating the followings displacement

            if(update)
            {
                if(_cclist_elem1[c1] != nullptr)
                    _cclist_elem1[c1]->setConstraintDForce(f, 3*c1, 3*c1+2, update);

                // set Delta force on object2 (if object2 exists)
                if(_cclist_elem2[c1] != nullptr)
                    _cclist_elem2[c1]->setConstraintDForce(f, 3*c1, 3*c1+2, update);
            }

            // set force on the contact force vector
            helper::set3Dof(f,c1,fn,ft,fs);
        }

        residuals->push_back(error);


        if (error < _tol*(numContacts+1))
        {
            msg_info() << "convergence after "<<it<<" iterations - error " << error;

            sofa::helper::AdvancedTimer::valSet("GS iterations", it+1);
            return 1;
        }

    }

    sofa::helper::AdvancedTimer::valSet("GS iterations", it);

    dmsg_warning() << "No convergence in unbuilt nlcp gaussseidel function : error ="
                <<error <<" after "<< it<<" iterations";

    return 0;
}

int LCPConstraintSolver::gaussseidel_unbuilt(SReal *dfree, SReal *f, std::vector<SReal>* residuals)
{
    const auto _mu = mu.getValue();

    if (_mu == 0.0)
        return lcp_gaussseidel_unbuilt(dfree, f, residuals);
    return nlcp_gaussseidel_unbuilt(dfree, f, residuals);
}



int LCPConstraintSolver::lcp_gaussseidel_unbuilt(SReal *dfree, SReal *f, std::vector<SReal>* /*residuals*/)
{
    if (!_numConstraints)
        return 0;

    auto buildConstraintsTimer = std::make_unique<sofa::helper::ScopedAdvancedTimer>("build_constraints");

    const auto _mu = mu.getValue();

    if(_mu!=0.0)
    {
        dmsg_warning() <<"friction case with unbuilt lcp is not implemented" ;
        return 0;
    }

    const int numContacts =  _numConstraints;
    int it,c1;

    // data for iterative procedure
    const SReal _tol = tol.getValue();
    const int _maxIt = maxIt.getValue();

    // indirection of the sequence of contact
    std::list<unsigned int> contact_sequence;

    for (unsigned int c=0; c< _numConstraints; c++)
    {
        contact_sequence.push_back(c);
    }


    for (unsigned int i=0; i<l_constraintCorrections.size(); i++)
    {
        core::behavior::BaseConstraintCorrection* cc = l_constraintCorrections[i];
        cc->resetForUnbuiltResolution(f, contact_sequence);
    }

    buildConstraintsTimer.reset();
    auto linkConstraintsTimer = std::make_unique<sofa::helper::ScopedAdvancedTimer>("link_constraints");

    //////// Important component if the LCP is not build :
    // for each contact, the pair of constraint correction that is involved with the contact is memorized
    _cclist_elem1.resize(numContacts);
    _cclist_elem2.resize(numContacts);
    std::vector<int> missingConstraintCorrectionContacts;
    for (c1=0; c1<numContacts; c1++)
    {
        bool elem1 = false;
        bool elem2 = false;
        for (const auto& cc : l_constraintCorrections)
        {
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
        if (!elem1)
        {
            _cclist_elem1[c1] = nullptr;
            missingConstraintCorrectionContacts.push_back(c1);
        }
        if(!elem2)
            _cclist_elem2[c1] = nullptr;
    }

    if (!missingConstraintCorrectionContacts.empty())
    {
        std::stringstream ss;
        for (const auto c : missingConstraintCorrectionContacts)
        {
            ss << c << ' ';
        }
        msg_error() << "The following contacts do not have an associated constraint correction component: " << ss.str();
        if (missingConstraintCorrectionContacts.size() == _cclist_elem1.size())
        {
            msg_error() << "None of the contacts has an associated constraint correction component: constraint correction is aborted";
            return 0;
        }
    }

    unbuilt_d.resize(_numConstraints);
    SReal *d = &(unbuilt_d[0]);

    linkConstraintsTimer.reset();
    auto buildDiagonalTimer = std::make_unique<sofa::helper::ScopedAdvancedTimer>("build_diagonal");

    //////////////
    // Beginning of iterative computations
    //////////////

    // the 1x1 diagonal block matrix is built:
    // for each contact, the pair of constraintcorrection is called to add the contribution
    for (c1=0; c1<numContacts; c1++)
    {
        // compliance of object1
        if (_cclist_elem1[c1] != nullptr)
        {
            _cclist_elem1[c1]->getBlockDiagonalCompliance(&_Wdiag, c1, c1);
        }
        // compliance of object2 (if object2 exists)
        if(_cclist_elem2[c1] != nullptr)
        {
            _cclist_elem2[c1]->getBlockDiagonalCompliance(&_Wdiag, c1, c1);
        }
    }

    unbuilt_W11.resize(numContacts);
    SReal *W11 = &(unbuilt_W11[0]);
    for (c1=0; c1<numContacts; c1++)
    {
        W11[c1] = _Wdiag.element(c1, c1);
    }

    buildDiagonalTimer.reset();
    SCOPED_TIMER_VARNAME(gaussSeidelTimer, "GAUSS_SEIDEL");

    SReal error = 0;
    SReal dn, fn, fn0;

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
            if (_cclist_elem1[c1] != nullptr)
            {
                _cclist_elem1[c1]->addConstraintDisplacement(d, c1, c1);
            }
            // displacement of object2 due to contact force (if object2 exists)
            if(_cclist_elem2[c1] != nullptr)
                _cclist_elem2[c1]->addConstraintDisplacement(d, c1, c1);
            // set displacement in dn
            dn=d[c1];

            // compute a new state for stick/slip
            /// ATTENTION  NOUVEAU GS_STATE : maintenant dn inclue les forces fn
            //W33[c1].New_GS_State(_mu,dn,dt,ds,fn,ft,fs);
            fn -= dn / W11[c1];

            if (fn < 0) fn = 0;
            error += fabs(W11[c1] * (fn - fn0));

            const bool update = (fn0 != 0.0 || fn != 0.0);

            if(update)
            {
                // set the new force :
                // compute the Delta of contact forces:
                f[c1] = fn - fn0;
                if (_cclist_elem1[c1] != nullptr)
                {
                    _cclist_elem1[c1]->setConstraintDForce(f, c1, c1, update);
                }
                if(_cclist_elem2[c1] != nullptr)
                    _cclist_elem2[c1]->setConstraintDForce(f, c1, c1, update);
            }

            f[c1] = fn;
        }

        if (error < _tol*(numContacts+1))
        {
            msg_info()<<"convergence after "<<it<<" iterations - error = " << error;
            sofa::helper::AdvancedTimer::valSet("GS iterations", it+1);

            return 1;
        }
    }

    sofa::helper::AdvancedTimer::valSet("GS iterations", it);

    dmsg_warning() <<" No convergence in  unbuilt lcp gaussseidel function : error ="
                <<error <<" after "<< it<<" iterations";

    return 0;
}

ConstraintProblem* LCPConstraintSolver::getConstraintProblem()
{
    return last_cp;
}

void LCPConstraintSolver::lockConstraintProblem(sofa::core::objectmodel::BaseObject* /*from*/, ConstraintProblem* l1, ConstraintProblem* l2)
{
    if((current_cp!=l1)&&(current_cp!=l2)) // Le lcp courant n'est pas locké
        return;

    if((&lcp1!=l1)&&(&lcp1!=l2)) // lcp1 n'est pas locké
        current_cp = &lcp1;
    else if((&lcp2!=l1)&&(&lcp2!=l2)) // lcp2 n'est pas locké
        current_cp = &lcp2;
    else
        current_cp = &lcp3; // lcp1 et lcp2 sont lockés, donc lcp3 n'est pas locké

    // Mise �  jour de _W _dFree et _result
    _W = &current_cp->W;
    _dFree = &current_cp->dFree;
    _result = &current_cp->f;
}


void LCPConstraintSolver::draw(const core::visual::VisualParams* vparams)
{
    unsigned int showLevels = (unsigned int) this->showLevels.getValue();
    if (showLevels > hierarchy_constraintBlockInfo.size()) showLevels = hierarchy_constraintBlockInfo.size();
    if (!showLevels) return;
    const SReal showCellWidth = this->showCellWidth.getValue();
    const type::Vec3 showTranslation = this->showTranslation.getValue();
    const type::Vec3 showLevelTranslation = this->showLevelTranslation.getValue();

    const int merge_spatial_step = this->merge_spatial_step.getValue();
    constexpr int merge_spatial_shift = 0; // merge_spatial_step/2
    const int merge_local_levels = this->merge_local_levels.getValue();
    const auto _mu = mu.getValue();

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    // from http://colorexplorer.com/colormatch.aspx
    const unsigned int colors[72]= { 0x2F2FBA, 0x111145, 0x2FBA8C, 0x114534, 0xBA8C2F, 0x453411, 0x2F72BA, 0x112A45,
        0x2FBA48, 0x11451B, 0xBA2F5B, 0x451122, 0x2FB1BA, 0x114145, 0x79BA2F, 0x2D4511, 0x9E2FBA, 0x3B1145, 0x2FBA79, 
        0x11452D, 0xBA662F, 0x452611, 0x2F41BA, 0x111845, 0x2FBA2F, 0x114511, 0xBA2F8C, 0x451134, 0x2F8CBA, 0x113445, 
        0x6DBA2F, 0x284511, 0xAA2FBA, 0x3F1145, 0x2FAABA, 0x113F45, 0xAFBA2F, 0x414511, 0x692FBA, 0x271145, 0x2FBAAA, 
        0x11453F, 0xBA892F, 0x453311, 0x2F31BA, 0x111245, 0x2FBA89, 0x114533, 0xBA4F2F, 0x451D11, 0x2F4DBA, 0x111C45, 
        0x2FBA6D, 0x114528, 0xBA2F56, 0x451120, 0x2F72BA, 0x112A45, 0x2FBA48, 0x11451B, 0xBA2F9A, 0x451139, 0x2F93BA, 
        0x113645, 0x3FBA2F, 0x174511, 0x662FBA, 0x261145, 0x2FBAA8, 0x11453E, 0xB1BA2F, 0x414511};

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
                    msg_info() << "Level " << level << ": constraint " << idFine << " from " << (info.parent ? info.parent->getName() : std::string("nullptr")) << " has invalid position index" ;
                    break;
                }
                if ((unsigned)(info.offsetDirection + 3*c) >= constraintDirections.size())
                {
                    msg_info() << "Level " << level << ": constraint " << idFine << " from " << (info.parent ? info.parent->getName() : std::string("nullptr")) << " has invalid direction index" ;
                    break;
                }
                ConstCoord posFine = constraintPositions[info.offsetPosition + c];
                ConstDeriv dirFineN  = constraintDirections[info.offsetDirection + 3*c + 0];
                ConstDeriv dirFineT1 = constraintDirections[info.offsetDirection + 3*c + 1];
                ConstDeriv dirFineT2 = constraintDirections[info.offsetDirection + 3*c + 2];
                const ConstArea area = (info.hasArea) ? constraintAreas[info.offsetArea + c] : (ConstArea)(2*coordFact*coordFact*showCellWidth*showCellWidth);

                type::Vec3 centerFine = showTranslation + showLevelTranslation*level;
                for (int i=0; i<3; ++i) centerFine[i] += ((posFine[i]+0.5)*coordFact + coord0) * showCellWidth;
                const SReal radius = sqrt(area*0.5);

                const unsigned int colid = (level * 12 + ((int)level < merge_local_levels ? (cb % 2) : 0)) % 72;
                color.i = (int) colors[colid + 0];
                vparams->drawTool()->drawArrow(
                    centerFine,centerFine+dirFineN*radius*2.0f,
                    (float)radius*2.0f*0.03f,
                    sofa::type::RGBAColor(
                            (float)(color.b[0]) * (1.0f/255.0f),
                            (float)(color.b[1]) * (1.0f/255.0f),
                            (float)(color.b[2]) * (1.0f/255.0f),
                            1.0f));
                if (_mu > 1.0e-6)
                {
                    color.i = (int) colors[colid + 2];
                    vparams->drawTool()->drawArrow(
                        centerFine-dirFineT1*radius*_mu,centerFine+dirFineT1*radius*_mu,
                        (float)(radius*_mu*0.03f),
                        sofa::type::RGBAColor(
                                (float)(color.b[0]) * (1.0f/255.0f),
                                (float)(color.b[1]) * (1.0f/255.0f),
                                (float)(color.b[2]) * (1.0f/255.0f),
                                1.0f));
                    color.i = (int) colors[colid + 4];
                    vparams->drawTool()->drawArrow(
                        centerFine-dirFineT2*radius*_mu,centerFine+dirFineT2*radius*_mu,
                        (float)(radius*_mu*0.03f),
                        sofa::type::RGBAColor(
                                color.b[0] * (1.0f/255.0f),
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

} //namespace sofa::component::constraint::lagrangian::solver
