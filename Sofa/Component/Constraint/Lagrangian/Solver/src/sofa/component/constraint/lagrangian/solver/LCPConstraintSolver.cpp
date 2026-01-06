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
    : d_displayDebug(initData(&d_displayDebug, false, "displayDebug", "Display debug information."))
    , d_initial_guess(initData(&d_initial_guess, true, "initial_guess", "activate LCP results history to improve its resolution performances."))
    , d_build_lcp(initData(&d_build_lcp, true, "build_lcp", "LCP is not fully built to increase performance in some case."))
    , d_tol(initData(&d_tol, 0.001_sreal, "tolerance", "residual error threshold for termination of the Gauss-Seidel algorithm"))
    , d_maxIt(initData(&d_maxIt, 1000, "maxIt", "maximal number of iterations of the Gauss-Seidel algorithm"))
    , d_regularizationTerm(initData(&d_regularizationTerm, 0.0_sreal, "regularizationTerm", "Add regularization factor times the identity matrix to the compliance W when solving constraints"))
    , d_mu(initData(&d_mu, 0.6_sreal, "mu", "Friction coefficient"))
    , d_minW(initData(&d_minW, 0.0_sreal, "minW", "If not zero, constraints whose self-compliance (i.e. the corresponding value on the diagonal of W) is smaller than this threshold will be ignored"))
    , d_maxF(initData(&d_maxF, 0.0_sreal, "maxF", "If not zero, constraints whose response force becomes larger than this threshold will be ignored"))
    , d_constraintForces(initData(&d_constraintForces,"constraintForces","OUTPUT: constraint forces (stored only if computeConstraintForces=True)"))
    , d_computeConstraintForces(initData(&d_computeConstraintForces,false,
                                        "computeConstraintForces",
                                        "enable the storage of the constraintForces."))
    , d_constraintGroups(initData(&d_constraintGroups, "group", "list of ID of groups of constraints to be handled by this solver."))
    , d_graph(initData(&d_graph, "graph", "Graph of residuals at each iteration"))
    , d_showCellWidth(initData(&d_showCellWidth, "showCellWidth", "Distance between each constraint cells"))
    , d_showTranslation(initData(&d_showTranslation, "showTranslation", "Position of the first cell"))
    , current_cp(&lcp1)
    , last_cp(nullptr)
    , _W(&lcp1.W)
    , _dFree(&lcp1.dFree)
    , _result(&lcp1.f)
{
    _numConstraints = 0;
    d_constraintGroups.beginEdit()->insert(0);
    d_constraintGroups.endEdit();

    d_graph.setWidget("graph");

    d_tol.setRequired(true);
    d_maxIt.setRequired(true);
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
    MechanicalVOpVisitor(core::execparams::defaultInstance(), (core::VecId)core::vec_id::write_access::dx).setMapped(true).execute( getContext()); //dX=0

    msg_info() <<" propagate DXn performed - collision called" ;

    SCOPED_TIMER("resetContactForce");
  
    for (const auto& cc : l_constraintCorrections)
    {
        cc->resetContactForce();
    }

    return true;
}


void LCPConstraintSolver::addRegularization(linearalgebra::BaseMatrix& W)
{
    const SReal regularization =  d_regularizationTerm.getValue();
    if (regularization>std::numeric_limits<SReal>::epsilon())
    {
        for (int i=0; i<W.rowSize(); ++i)
        {
            W.add(i,i,regularization);
        }
    }
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

    cparams.setX(core::vec_id::read_access::freePosition);
    cparams.setV(core::vec_id::read_access::freeVelocity);

    _numConstraints = buildConstraintMatrix(&cparams);
    sofa::helper::AdvancedTimer::valSet("numConstraints", _numConstraints);

    current_cp->mu = d_mu.getValue();
    current_cp->clear(_numConstraints);

    getConstraintViolation(&cparams, _dFree);

    if (d_build_lcp.getValue())
    {
        addComplianceInConstraintSpace(cparams);
    }
    else
    {
        // When the LCP or the NLCP is not fully built, the  diagonal blocks of the matrix are still needed for the resolution
        _Wdiag.resize(_numConstraints,_numConstraints);
    }

    m_constraintBlockInfo.clear();
    m_constraintIds.clear();

    getConstraintInfo(cparams);

}

bool LCPConstraintSolver::solveSystem(const core::ConstraintParams * /*cParams*/, MultiVecId /*res1*/, MultiVecId /*res2*/)
{
    const auto _mu = d_mu.getValue();

    std::map < std::string, sofa::type::vector<SReal> >& graph = *d_graph.beginEdit();

    if (d_build_lcp.getValue())
    {
        const SReal _tol = d_tol.getValue();
        const int _maxIt = d_maxIt.getValue();
        const SReal _minW = d_minW.getValue();
        const SReal _maxF = d_maxF.getValue();

        if (_mu > 0.0)
        {
            current_cp->tolerance = _tol;

            sofa::type::vector<SReal>& graph_error = graph["Error"];
            graph_error.clear();
            sofa::type::vector<SReal>& graph_violations = graph["Violation"];
            graph_violations.clear();

            {
                SCOPED_TIMER("NLCP GaussSeidel");
                helper::nlcp_gaussseidel(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _mu, _tol, _maxIt, d_initial_guess.getValue(),
                                         notMuted(), _minW, _maxF, &graph_error, &graph_violations);
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

        if (d_displayDebug.getValue())
        {
            dmsg_info() <<"_result unbuilt:"<<(*_result) ;

            _result->resize(_numConstraints);

            const SReal _tol = d_tol.getValue();
            const int _maxIt = d_maxIt.getValue();

            buildSystem();

            helper::nlcp_gaussseidel(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _mu, _tol, _maxIt, d_initial_guess.getValue());
            dmsg_info() <<"\n_result nlcp :"<<(*_result);
        }
    }

    d_graph.endEdit();

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
    if (d_initial_guess.getValue())
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



void LCPConstraintSolver::getConstraintInfo(core::ConstraintParams cparams)
{
    if (d_initial_guess.getValue() && (_numConstraints != 0))
    {
        {
            SCOPED_TIMER("Get Constraint Info");
            MechanicalGetConstraintInfoVisitor(&cparams, m_constraintBlockInfo, m_constraintIds).execute(getContext());
        }
        if (d_initial_guess.getValue())
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


    addRegularization(* _W);
    dmsg_info() << "W=" << *_W ;
}

void LCPConstraintSolver::build_Coarse_Compliance(std::vector<int> &constraint_merge, int sizeCoarseSystem)
{
    /* constraint_merge => tableau donne l'indice du groupe de contraintes dans le système grossier en fonction de l'indice de la contrainte dans le système de départ */
    dmsg_info() <<"build_Coarse_Compliance is called : size="<<sizeCoarseSystem ;

    _Wcoarse.clear();

    dmsg_error_when(sizeCoarseSystem==0) <<"no constraint" ;

    _Wcoarse.resize(sizeCoarseSystem,sizeCoarseSystem);
    for (const auto& cc : l_constraintCorrections)
    {
        cc->getComplianceWithConstraintMerge(&_Wcoarse, constraint_merge);
    }
}

void LCPConstraintSolver::computeInitialGuess()
{
    sofa::helper::AdvancedTimer::StepVar vtimer("InitialGuess");

    const auto _mu = d_mu.getValue();
    const VecConstraintBlockInfo& constraintBlockInfo = m_constraintBlockInfo;
    const VecPersistentID& constraintIds = m_constraintIds;
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
        std::map<core::behavior::BaseLagrangianConstraint*, ConstraintBlockBuf>::const_iterator previt = _previousConstraints.find(info.parent);
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
    const VecConstraintBlockInfo& constraintBlockInfo = m_constraintBlockInfo;
    const VecPersistentID& constraintIds = m_constraintIds;
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

    auto _mu = d_mu.getValue();
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
    SReal _tol = d_tol.getValue();
    int _maxIt = d_maxIt.getValue();

    /// each constraintCorrection has an internal force vector that is set to "0"

    // indirection of the sequence of contact
    std::list<unsigned int> contact_sequence;

    for (unsigned int c=0; c< _numConstraints; c++)
    {
        contact_sequence.push_back(c);
    }


    for (const auto& cc : l_constraintCorrections)
    {
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

    addRegularization(_Wdiag);


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

            // set Delta force on object 1 for evaluating the following displacement

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
    const auto _mu = d_mu.getValue();

    if (_mu == 0.0)
        return lcp_gaussseidel_unbuilt(dfree, f, residuals);
    return nlcp_gaussseidel_unbuilt(dfree, f, residuals);
}



int LCPConstraintSolver::lcp_gaussseidel_unbuilt(SReal *dfree, SReal *f, std::vector<SReal>* /*residuals*/)
{
    if (!_numConstraints)
        return 0;

    auto buildConstraintsTimer = std::make_unique<sofa::helper::ScopedAdvancedTimer>("build_constraints");

    const auto _mu = d_mu.getValue();

    if(_mu!=0.0)
    {
        dmsg_warning() <<"friction case with unbuilt lcp is not implemented" ;
        return 0;
    }

    const int numContacts =  _numConstraints;
    int it,c1;

    // data for iterative procedure
    const SReal _tol = d_tol.getValue();
    const int _maxIt = d_maxIt.getValue();

    // indirection of the sequence of contact
    std::list<unsigned int> contact_sequence;

    for (unsigned int c=0; c< _numConstraints; c++)
    {
        contact_sequence.push_back(c);
    }


    for (const auto& cc : l_constraintCorrections)
    {
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

    addRegularization(_Wdiag);

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


void registerLCPConstraintSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("A Constraint Solver using the Linear Complementarity Problem formulation to solve BaseConstraint based components.")
        .add< LCPConstraintSolver >());
}

} //namespace sofa::component::constraint::lagrangian::solver
