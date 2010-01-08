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

#include <sofa/component/constraint/LCPConstraintSolver.h>

#include <sofa/simulation/common/AnimateVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/SolveVisitor.h>

#include <sofa/helper/system/thread/CTime.h>
#include <math.h>
#include <iostream>

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace constraint
{


LCP::LCP(unsigned int mxC) : maxConst(mxC), tol(0.00001), numItMax(1000), useInitialF(true), mu(0.0), dim(0), lok(false)
{
    W.resize(maxConst,maxConst);
    dFree.resize(maxConst);
    f.resize(2*maxConst+1);
}

LCP::~LCP()
{
}

void LCP::reset(void)
{
    W.clear();
    W.clear();
    dFree.clear();
}


bool LCPConstraintSolver::prepareStates(double /*dt*/, VecId id)
{
    if (id != VecId::freePosition()) return false;


    last_lcp = lcp;
    core::componentmodel::behavior::BaseMechanicalState::VecId dx_id = core::componentmodel::behavior::BaseMechanicalState::VecId::dx();
    simulation::MechanicalVOpVisitor(dx_id).execute( context); //dX=0
    simulation::MechanicalPropagateDxVisitor(dx_id,true).execute( context); //Propagate dX //ignore the mask here

    if( f_printLog.getValue())
        serr<<" propagate DXn performed - collision called"<<sendl;

    time = 0.0;
    timeTotal=0.0;
    timeScale = 1000.0 / (double)CTime::getTicksPerSec();

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }

    if ( displayTime.getValue() )
    {
        time = (double) timer.getTime();
        timeTotal = (double) timerTotal.getTime();
    }
    return true;
}
bool LCPConstraintSolver::buildSystem(double /*dt*/, VecId)
{
    //sout<<"constraintCorrections is called"<<sendl;

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
bool LCPConstraintSolver::solveSystem(double /*dt*/, VecId)
{
    if(build_lcp.getValue())
    {

        double _tol = tol.getValue();
        int _maxIt = maxIt.getValue();
        if (_mu > 0.0)
        {

            lcp->setNbConst(_numConstraints);
            lcp->setTol(_tol);
            std::cout<<"+++++++++++++ \n SOLVE WITH MULTIGRID \n ++++++++++++++++"<<std::endl;

            MultigridConstraintsMerge();
            //build_Coarse_Compliance(_constraint_group, 3*_group_lead.size());
            std::cerr<<"out from build_Coarse_Compliance"<<std::endl;


            if (multi_grid.getValue())
            {
                helper::nlcp_multiGrid_2levels(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _mu, _tol, _maxIt, initial_guess.getValue(),
                        _contact_group, _group_lead.size(), this->f_printLog.getValue());
                std::cout<<"+++++++++++++ \n SOLVE WITH GAUSSSEIDEL \n ++++++++++++++++"<<std::endl;
                helper::nlcp_gaussseidel(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _mu, _tol, _maxIt, initial_guess.getValue());
                if (this->f_printLog.getValue()) helper::afficheLCP(_dFree->ptr(), _W->lptr(), _result->ptr(),_numConstraints);

            }
            else
                helper::nlcp_gaussseidel(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _mu, _tol, _maxIt, initial_guess.getValue());



        }
        else
        {
            // warning _A has been being suppr... need to be allocated
            //
            //		helper::lcp_lexicolemke(_numConstraints, _dFree->ptr(), _W->lptr(), _A.lptr(), _result->ptr());
            helper::gaussSeidelLCP1(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _tol, _maxIt);
            if (this->f_printLog.getValue()) helper::afficheLCP(_dFree->ptr(), _W->lptr(), _result->ptr(),_numConstraints);
        }
    }
    else
    {

        //std::cout<<"gaussseidel_unbuilt"<<std::endl;
        //std::cout<<"_result-before :"<<_result<<std::endl;
        gaussseidel_unbuilt(_dFree->ptr(), _result->ptr());
        //std::cout<<"\n_result unbuilt:"<<(*_result)<<std::endl;

        /////// debug
        /*
           _result->resize(_numConstraints);

           double _tol = tol.getValue();
           int _maxIt = maxIt.getValue();

           build_LCP();
           helper::nlcp_gaussseidel(_numConstraints, _dFree->ptr(), _W->lptr(), _result->ptr(), _mu, _tol, _maxIt, initial_guess.getValue());
           std::cout<<"\n_result nlcp :"<<(*_result)<<std::endl;
        */
        //std::cout<<"LCP:"<<std::endl;
        //helper::afficheLCP(_dFree->ptr(), _W->lptr(), _result->ptr(),_numConstraints);
        //std::cout<<"build_problem_info is called"<<std::endl;

        ////////

    }

    if ( displayTime.getValue() )
    {
        sout<<" TOTAL solve_LCP " <<( (double) timer.getTime() - time)*timeScale<<" ms" <<sendl;
        time = (double) timer.getTime();
    }
    return true;
}
bool LCPConstraintSolver::applyCorrection(double /*dt*/, VecId )
{
    if (initial_guess.getValue())
        keepContactForcesValue();

    if(this->f_printLog.getValue())
    {
        serr<<"keepContactForces done"<<sendl;

    }

    //	MechanicalApplyContactForceVisitor(_result).execute(context);
    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->applyContactForce(_result);
    }
    if(this->f_printLog.getValue())
    {
        serr<<"applyContactForce in constraintCorrection done"<<sendl;

    }

    simulation::MechanicalPropagateAndAddDxVisitor().execute( context);

    if(this->f_printLog.getValue())
    {
        serr<<"propagate corrective motion done"<<sendl;

    }

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }

    if (displayTime.getValue())
    {
        sout<<" TotalTime " <<( (double) timerTotal.getTime() - timeTotal)*timeScale <<" ms" <<sendl;
    }
    return true;
}









#define MAX_NUM_CONSTRAINTS 3000
//#define DISPLAY_TIME

LCPConstraintSolver::LCPConstraintSolver()
    :displayTime(initData(&displayTime, false, "displayTime","Display time for each important step of LCPConstraintSolver."))
    ,initial_guess(initData(&initial_guess, true, "initial_guess","activate LCP results history to improve its resolution performances."))
    ,build_lcp(initData(&build_lcp, true, "build_lcp", "LCP is not fully built to increase performance in some case."))
    ,multi_grid(initData(&multi_grid, false, "multi_grid","activate multi_grid resolution (NOT STABLE YET)"))
    ,tol( initData(&tol, 0.001, "tolerance", ""))
    ,maxIt( initData(&maxIt, 1000, "maxIt", ""))
    ,mu( initData(&mu, 0.6, "mu", ""))
    , constraintGroups( initData(&constraintGroups, "group", "list of ID of groups of constraints to be handled by this solver.") )
    ,_mu(0.6)
    , lcp1(MAX_NUM_CONSTRAINTS)
    , lcp2(MAX_NUM_CONSTRAINTS)
    , lcp3(MAX_NUM_CONSTRAINTS)
    , _W(&lcp1.W)
    , lcp(&lcp1)
    ,last_lcp(0)
    , _dFree(&lcp1.dFree)
    , _result(&lcp1.f)
{
    _numConstraints = 0;
    _mu = 0.0;
    constraintGroups.beginEdit()->insert(0);
    constraintGroups.endEdit();

    //_numPreviousContact=0;
    //_PreviousContactList = (contactBuf *)malloc(MAX_NUM_CONSTRAINTS * sizeof(contactBuf));
    //_cont_id_list = (long *)malloc(MAX_NUM_CONSTRAINTS * sizeof(long));

    _Wdiag = new SparseMatrix<double>();

}

void LCPConstraintSolver::init()
{
    core::componentmodel::behavior::ConstraintSolver::init();

    // Prevents ConstraintCorrection accumulation due to multiple MasterSolver initialization on dynamic components Add/Remove operations.
    if (!constraintCorrections.empty())
    {
        constraintCorrections.clear();
    }

    getContext()->get<core::componentmodel::behavior::BaseConstraintCorrection>(&constraintCorrections, core::objectmodel::BaseContext::SearchDown);

    context = (simulation::Node*) getContext();
}

void LCPConstraintSolver::build_LCP()
{
    _numConstraints = 0;
    //sout<<" accumulateConstraint "  <<sendl;

    // mechanical action executed from root node to propagate the constraints
    simulation::MechanicalResetConstraintVisitor().execute(context);
    simulation::MechanicalAccumulateConstraint(_numConstraints).execute(context);
    _mu = mu.getValue();


    //sout<<" accumulateConstraint_done "  <<sendl;

    if (_numConstraints > MAX_NUM_CONSTRAINTS)
    {
        serr<<sendl<<"Error in LCPConstraintSolver, maximum number of contacts exceeded, "<< _numConstraints/3 <<" contacts detected"<<sendl;
        exit(-1);
    }

    lcp->getMu() = _mu;

    _dFree->resize(_numConstraints);
    _W->resize(_numConstraints,_numConstraints);

    MechanicalGetConstraintValueVisitor(_dFree).execute(context);
    //	simulation::MechanicalComputeComplianceVisitor(_W).execute(context);

    if (this->f_printLog.getValue()) sout<<"LCPConstraintSolver: "<<_numConstraints<<" constraints, mu = "<<_mu<<sendl;

    //sout<<" computeCompliance in "  << constraintCorrections.size()<< " constraintCorrections" <<sendl;

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->getCompliance(_W);
    }
    //sout<<" computeCompliance_done "  <<sendl;

    if ((initial_guess.getValue()) && (_numConstraints != 0))
    {
        //_cont_id_list.resize(_numConstraints);
        //MechanicalGetContactIDVisitor(&(_cont_id_list[0])).execute(context);
        _constraintGroupInfo.clear();
        _constraintIds.clear();
        _constraintPositions.clear();
        MechanicalGetConstraintInfoVisitor(_constraintGroupInfo, _constraintIds, _constraintPositions).execute(context);
        computeInitialGuess();
    }
}

void LCPConstraintSolver::build_Coarse_Compliance(std::vector<int> &constraint_merge, int sizeCoarseSystem)
{
    /* constraint_merge => tableau donne l'indice du groupe de contraintes dans le système grossier en fonction de l'indice de la contrainte dans le système de départ */
    std::cout<<"build_Coarse_Compliance is called : size="<<sizeCoarseSystem<<std::endl;

    _Wcoarse.clear();
    if (sizeCoarseSystem==0)
    {
        std::cerr<<"no constraint"<<std::endl;
        return;
    }
    _Wcoarse.resize(sizeCoarseSystem,sizeCoarseSystem);
    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->getComplianceWithConstraintMerge(&_Wcoarse, constraint_merge);
    }
}

void LCPConstraintSolver::MultigridConstraintsMerge()
{
    /////// Analyse des contacts à regrouper //////
    double criterion=0.0;
    int numContacts = _numConstraints/3;
    _contact_group.clear();
    _contact_group.resize(numContacts);
    _group_lead.clear();
    _constraint_group.clear();
    _constraint_group.resize(_numConstraints);

    for (int c=0; c<numContacts; c++)
    {
        bool new_group = true;
        for(int g=0; g<(int)_group_lead.size() ; g++)
        {

            if (_W->lptr()[3*c][3*_group_lead[g]] > criterion * (_W->lptr()[3*c][3*c] +_W->lptr()[3*_group_lead[g]][3*_group_lead[g]]) )  // on regarde les couplages selon la normale...
            {
                new_group =false;
                _contact_group[c] = g;


            }
        }
        if (new_group)
        {
            _contact_group[c]=_group_lead.size();
            _group_lead.push_back(c);

        }
    }

    if(this->f_printLog.getValue())
    {
        std::cout<<"contacts merged in "<<_group_lead.size()<<" list(s)"<<std::endl;
    }

    for (int c=0; c<numContacts; c++)
    {
        _constraint_group[3*c] = 3*_contact_group[c];
        _constraint_group[3*c+1] = 3*_contact_group[c]+1;
        _constraint_group[3*c+2] = 3*_contact_group[c]+2;

    }
}


/// build_problem_info
/// When the LCP or the NLCP is not fully built, the  diagonal blocks of the matrix are still needed for the resolution
/// This function ask to the constraintCorrection classes to build this diagonal blocks
void LCPConstraintSolver::build_problem_info()
{

    // debug
    //std::cout<<" accumulateConstraint "  <<std::endl;
    _numConstraints = 0;
    // mechanical action executed from root node to propagate the constraints
    simulation::MechanicalResetConstraintVisitor().execute(context);
    simulation::MechanicalAccumulateConstraint(_numConstraints).execute(context);
    _mu = mu.getValue();

    // debug
    //std::cout<<" accumulateConstraint_done "  <<std::endl;

    // necessary ///////
    if (_numConstraints > MAX_NUM_CONSTRAINTS)
    {
        serr<<sendl<<"WARNING in LCPConstraintSolver: maximum number of contacts exceeded, "<< _numConstraints/3 <<" contacts detected"<<sendl;
        //exit(-1);
    }

    lcp->getMu() = _mu;

    _dFree->resize(_numConstraints);
    // as _Wdiag is a sparse matrix resize do not allocate memory
    _Wdiag->resize(_numConstraints,_numConstraints);
    _result->resize(_numConstraints);

    // debug
    //std::cout<<" resize done "  <<std::endl;

    MechanicalGetConstraintValueVisitor(_dFree).execute(context);

    if (this->f_printLog.getValue()) sout<<"LCPConstraintSolver: "<<_numConstraints<<" constraints, mu = "<<_mu<<sendl;

    //debug
    //std::cout<<" computeCompliance in "  << constraintCorrections.size()<< " constraintCorrections" <<std::endl;


    if (initial_guess.getValue())
    {
        _constraintGroupInfo.clear();
        _constraintIds.clear();
        _constraintPositions.clear();
        MechanicalGetConstraintInfoVisitor(_constraintGroupInfo, _constraintIds, _constraintPositions).execute(context);
        computeInitialGuess();
    }






}



void LCPConstraintSolver::computeInitialGuess()
{
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
    for (unsigned cg = 0; cg < _constraintGroupInfo.size(); ++cg)
    {
        const ConstraintGroupInfo& info = _constraintGroupInfo[cg];
        if (!info.hasId) continue;
        std::map<core::componentmodel::behavior::BaseConstraint*, ConstraintGroupBuf>::const_iterator previt = _previousConstraints.find(info.parent);
        if (previt == _previousConstraints.end()) continue;
        const ConstraintGroupBuf& buf = previt->second;
        const int c0 = info.const0;
        const int nbl = (info.nbLines < buf.nbLines) ? info.nbLines : buf.nbLines;
        for (int c = 0; c < info.nbGroups; ++c)
        {
            std::map<PersistentID,int>::const_iterator it = buf.persistentToConstraintIdMap.find(_constraintIds[info.offsetId + c]);
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
    // store current force
    _previousForces.resize(_numConstraints);
    for (unsigned int c=0; c<_numConstraints; ++c)
        _previousForces[c] = (*_result)[c];
    // clear previous history
    for (std::map<core::componentmodel::behavior::BaseConstraint*, ConstraintGroupBuf>::iterator it = _previousConstraints.begin(), itend = _previousConstraints.end(); it != itend; ++it)
    {
        ConstraintGroupBuf& buf = it->second;
        for (std::map<PersistentID,int>::iterator it2 = buf.persistentToConstraintIdMap.begin(), it2end = buf.persistentToConstraintIdMap.end(); it2 != it2end; ++it2)
            it2->second = -1;
    }
    // fill info from current ids
    for (unsigned cg = 0; cg < _constraintGroupInfo.size(); ++cg)
    {
        const ConstraintGroupInfo& info = _constraintGroupInfo[cg];
        if (!info.parent) continue;
        if (!info.hasId) continue;
        ConstraintGroupBuf& buf = _previousConstraints[info.parent];
        int c0 = info.const0;
        int nbl = info.nbLines;
        buf.nbLines = nbl;
        for (int c = 0; c < info.nbGroups; ++c)
            buf.persistentToConstraintIdMap[_constraintIds[info.offsetId + c]] = c0 + c*nbl;
    }
}


int LCPConstraintSolver::nlcp_gaussseidel_unbuilt(double *dfree, double *f)
{

    helper::system::thread::CTime timer;
    double time = 0.0;
    double timeScale = 1000.0 / (double)CTime::getTicksPerSec();
    if ( displayTime.getValue() )
    {
        time = (double) timer.getTime();
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

    // if necessary: modify the sequence of contact
    std::list<int> contact_sequence;

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetForUnbuiltResolution(f, contact_sequence);
    }

    // debug
    // std::cout<<"getBlockDiagonalCompliance  Wdiag = "<<(* _Wdiag)<<std::endl;
    // return 1;
    if ( displayTime.getValue() )
    {
        sout<<" build_constraints " << ( (double) timer.getTime() - time)*timeScale<<" ms" <<sendl;
        time = (double) timer.getTime();
    }

    bool change_contact_sequence = false;

    if(contact_sequence.size() ==_numConstraints)
        change_contact_sequence=true;


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

            core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
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
        //debug
        //std::cout<<"contact "<<c1<<" cclist_elem1 : "<<_cclist_elem1[c1]->getName()<<std::endl;
        // compliance of object1
        _cclist_elem1[c1]->getBlockDiagonalCompliance(_Wdiag, 3*c1, 3*c1+2);
        // compliance of object2 (if object2 exists)
        if(_cclist_elem2[c1] != NULL)
        {
            _cclist_elem2[c1]->getBlockDiagonalCompliance(_Wdiag, 3*c1, 3*c1+2);
            // debug
            //std::cout<<"_cclist_elem2[c1]"<<std::endl;
        }
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
    // std::cout<<"getBlockDiagonalCompliance  Wdiag = "<<(* _Wdiag)<<std::endl;
    // return 1;
    if ( displayTime.getValue() )
    {
        sout<<" build_diagonal " << ( (double) timer.getTime() - time)*timeScale<<" ms" <<sendl;
        time = (double) timer.getTime();
    }

    double error = 0;
    double dn, dt, ds, fn, ft, fs, fn0;

    for (it=0; it<_maxIt; it++)
    {
        std::list<int>::iterator it_c = contact_sequence.begin();
        error =0;
        for (int c=0; c<numContacts; c++)
        {
            if(change_contact_sequence)
            {
                int constraint = *it_c;
                c1 = constraint/3;
                it_c++; it_c++; it_c++;

            }
            else
                c1=c;

            //std::cout<<"it"<<it << " - c1 :"<<c1<<std::endl;

            // compute the current violation :

            // violation when no contact force
            d[3*c1]=dfree[3*c1]; d[3*c1+1]=dfree[3*c1+1]; d[3*c1+2]=dfree[3*c1+2];

            // debug
            //if(c1<2)
            //	std::cout<<"free displacement for contact : dn_free = "<< d[3*c1] <<"  dt_free = "<<d[3*c1+1]<<"  ds_free = "<<d[3*c1+2]<<std::endl;


            // set current force in fn, ft, fs
            fn0=fn=f[3*c1]; ft=f[3*c1+1]; fs=f[3*c1+2];
            //f[3*c1] = 0.0; f[3*c1+1] = 0.0; f[3*c1+2] = 0.0;

            // displacement of object1 due to contact force
            _cclist_elem1[c1]->addConstraintDisplacement(d, 3*c1, 3*c1+2);

            // displacement of object2 due to contact force (if object2 exists)
            if(_cclist_elem2[c1] != NULL)
                _cclist_elem2[c1]->addConstraintDisplacement(d, 3*c1, 3*c1+2);


            // set displacement in dn, dt, ds
            dn=d[3*c1]; dt=d[3*c1+1]; ds=d[3*c1+2];
            //d[3*c1  ] = dn + (W33[c1].w[0]*fn + W33[c1].w[1]*ft + W33[c1].w[2]*fs);
            //d[3*c1+1] = dt + (W33[c1].w[1]*fn + W33[c1].w[3]*ft + W33[c1].w[4]*fs);
            //d[3*c1+2] = ds + (W33[c1].w[2]*fn + W33[c1].w[4]*ft + W33[c1].w[5]*fs);

            // debug
            //if(c1<2)
            //	std::cout<<"New_GS_State called : dn = "<<dn<<"  dt = "<<dt<<"  ds = "<<ds<<"  fn = "<<fn<<"  ft = "<<ft<<"  fs = "<<fs<<std::endl;





            // compute a new state for stick/slip
            /// ATTENTION  NOUVEAU GS_STATE : maintenant dn, dt et ds inclue les forces fn, ft, fs
            W33[c1].New_GS_State(_mu,dn,dt,ds,fn,ft,fs);
            //W33[c1].GS_State(_mu,dn,dt,ds,fn,ft,fs);
            // debug
            //if(c1<2)
            //	std::cout<<"New_GS_State solved for contact "<<c1<<" : dn = "<<dn<<"  dt = "<<dt<<"  ds = "<<ds<<"  fn = "<<fn<<"  ft = "<<ft<<"  fs = "<<fs<<std::endl;

            // evaluate an error (based on displacement)
            error += helper::absError(dn,dt,ds,d[3*c1],d[3*c1+1],d[3*c1+2]);




            bool update;
            if (fn0 == 0.0 && fn == 0.0)
                update=false;
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


            ///// debug : verifie si on retrouve le mÃªme dn
            /*
              d[3*c1]=dfree[3*c1]; d[3*c1+1]=dfree[3*c1+1]; d[3*c1+2]=dfree[3*c1+2];
              _cclist_elem1[c1]->addConstraintDisplacement(d, 3*c1, 3*c1+2);
              if(fabs(dn-d[3*c1]) > 0.000000001*fabs(dn) && dn> 0.1*_tol)
              std::cerr<<"WARNING debug : dn ="<<dn<<" d["<<3*c1<<"]= "<< d[3*c1]<<" dfree= "<<dfree[3*c1]<<"  - update :"<<update<<" with fn ="<<fn<<" and f["<<3*c1<<"]= "<< fn-f[3*c1  ]<<std::endl;
            */

            // set force on the contact force vector
            helper::set3Dof(f,c1,fn,ft,fs);

        }

        if (error < _tol*(numContacts+1))
        {
            //free(d);
            if ( displayTime.getValue() )
            {
                sout<<"convergence after "<<it<<" iterations - error"<<error<<sendl;
            }
            //debug
            //std::cout<<" f : ["<<std::endl;
            for (int i = 0; i < numContacts; i++)
            {
                //	std::cout<<f[3*i]<<"\n"<<f[3*i+1] <<"\n"<<f[3*i+2] <<std::endl;
            }
            //std::cout<<"];"<<std::endl;
            //delete[] W33;
            if ( displayTime.getValue() )
            {
                sout<<" GAUSS_SEIDEL iterations  " << ( (double) timer.getTime() - time)*timeScale<<" ms" <<sendl;

            }


            return 1;
        }
    }
    //free(d);
    //for (int i = 0; i < numContacts; i++)
    //	delete W33[i];
    //delete[] W33;
    if ( displayTime.getValue() )
    {
        sout<<" GAUSS_SEIDEL iterations  " << ( (double) timer.getTime() - time)*timeScale<<" ms" <<sendl;
    }

    std::cerr<<"\n No convergence in  unbuilt nlcp gaussseidel function : error ="<<error <<" after"<< it<<" iterations"<<std::endl;
    //afficheLCP(dfree,W,f,dim);
    return 0;




}


int LCPConstraintSolver::lcp_gaussseidel_unbuilt(double *dfree, double *f)
{
    helper::system::thread::CTime timer;
    double time = 0.0;
    double timeScale = 1.0;
    if ( displayTime.getValue() )
    {
        time = (double) timer.getTime();
        timeScale = 1000.0 / (double)CTime::getTicksPerSec();
    }


    if(_mu!=0.0)
    {
        serr<<"WARNING: friction case with unbuilt lcp is not implemented"<<sendl;
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

    // if necessary: modify the sequence of contact
    std::list<int> contact_sequence;

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetForUnbuiltResolution(f, contact_sequence);
    }

    if ( displayTime.getValue() )
    {
        sout<<" build_constraints " << ( (double) timer.getTime() - time)*timeScale<<" ms" <<sendl;
        time = (double) timer.getTime();
    }

    bool change_contact_sequence = false;

    if(contact_sequence.size() ==_numConstraints)
        change_contact_sequence=true;


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

            core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
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
        if(!elem1)
            serr<<"WARNING: no constraintCorrection found for contact"<<c1<<sendl;
        if(!elem2)
            _cclist_elem2[c1] = (NULL);
    }

    unbuilt_d.resize(_numConstraints);
    double *d = &(unbuilt_d[0]);

    if ( displayTime.getValue() )
    {
        sout<<" link_constraints " << ( (double) timer.getTime() - time)*timeScale<<" ms" <<sendl;
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
        sout<<" build_diagonal " << ( (double) timer.getTime() - time)*timeScale<<" ms" <<sendl;
        time = (double) timer.getTime();
    }

    double error = 0;
    double dn, fn, fn0;

    for (it=0; it<_maxIt; it++)
    {
        std::list<int>::iterator it_c = contact_sequence.begin();
        error =0;
        for (int c=0; c<numContacts; c++)
        {
            if(change_contact_sequence)
            {
                int constraint = *it_c;
                c1 = constraint;
                it_c++;

            }
            else
                c1=c;

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
                sout<<"convergence after "<<it<<" iterations - error = "<<error<<sendl;
                sout<<" GAUSS_SEIDEL iterations  " << ( (double) timer.getTime() - time)*timeScale<<" ms" <<sendl;

            }
            return 1;
        }
    }
    if ( displayTime.getValue() )
    {
        sout<<" GAUSS_SEIDEL iterations " << ( (double) timer.getTime() - time)*timeScale<<" ms" <<sendl;
    }

    serr<<"No convergence in  unbuilt lcp gaussseidel function : error ="<<error <<" after"<< it<<" iterations"<<sendl;
    //afficheLCP(dfree,W,f,dim);
    return 0;
}

LCP* LCPConstraintSolver::getLCP()
{
    return last_lcp;
}

void LCPConstraintSolver::lockLCP(LCP* l1, LCP* l2)
{
    if((lcp!=l1)&&(lcp!=l2)) // Le lcp courrant n'est pas locké
        return;

    if((&lcp1!=l1)&&(&lcp1!=l2)) // lcp1 n'est pas locké
        lcp = &lcp1;
    else if((&lcp2!=l1)&&(&lcp2!=l2)) // lcp2 n'est pas locké
        lcp = &lcp2;
    else
        lcp = &lcp3; // lcp1 et lcp2 sont lockés, donc lcp3 n'est pas locké

    // Mise à jour de _W _dFree et _result
    _W = &lcp->W;
    _dFree = &lcp->dFree;
    _result = &lcp->f;
}





int LCPConstraintSolverClass = core::RegisterObject("A Constraint Solver using the Linear Complementarity Problem formulation to solve BaseConstraint based components")
        .add< LCPConstraintSolver >();

SOFA_DECL_CLASS(LCPConstraintSolver);


} // namespace constraint

} // namespace component

} // namespace sofa
