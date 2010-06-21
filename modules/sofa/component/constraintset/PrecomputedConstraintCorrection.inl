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
#ifndef SOFA_CORE_COLLISION_CONTACTCORRECTION_INL
#define SOFA_CORE_COLLISION_CONTACTCORRECTION_INL

#include "PrecomputedConstraintCorrection.h"
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/MechanicalVisitor.h>

#include <sofa/component/odesolver/EulerImplicitSolver.h>

#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>

#include <sofa/component/container/RotationFinder.h>

#include <sofa/helper/gl/DrawManager.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/Quater.h>

#ifdef SOFA_HAVE_EIGEN2
#include <sofa/component/constraintset/LMConstraintSolver.h>
#include <sofa/simulation/common/Node.h>
#endif

//#include <glib.h>
#include <sstream>
#include <list>
#include <iomanip>

//#define NEW_METHOD_UNBUILT

namespace sofa
{

namespace component
{

namespace constraintset
{

//#define	MAX_NUM_CONSTRAINT_PER_NODE 10000
//#define EPS_UNITARY_FORCE 0.01

using namespace sofa::component::odesolver;
using namespace sofa::component::linearsolver;
using namespace sofa::simulation;

template<class DataTypes>
PrecomputedConstraintCorrection<DataTypes>::PrecomputedConstraintCorrection(behavior::MechanicalState<DataTypes> *mm)
    : _rotations(false)
    , f_rotations(initDataPtr(&f_rotations,&_rotations,"rotations",""))
    , _restRotations(false)
    , f_restRotations(initDataPtr(&f_restRotations,&_restRotations,"restDeformations",""))
    , recompute(initData(&recompute, false, "recompute","if true, always recompute the compliance"))
//	, filePrefix(initData(&filePrefix, "filePrefix","if not empty, the prefix used for the file containing the compliance matrix"))
    , debugViewFrameScale(initData(&debugViewFrameScale, 1.0, "debugViewFrameScale","Scale on computed node's frame"))
    , f_fileCompliance(initData(&f_fileCompliance, "fileCompliance", "Precomputed compliance matrix data file"))
    , mstate(mm)
    , invM(NULL)
    , appCompliance(NULL)
    , nbRows(0), nbCols(0), dof_on_node(0), nbNodes(0)
{
    addAlias(&f_fileCompliance, "filePrefix");
}

template<class DataTypes>
PrecomputedConstraintCorrection<DataTypes>::~PrecomputedConstraintCorrection()
{
    releaseInverse(invName, invM);
}



//////////////////////////////////////////////////////////////////////////
//   Precomputation of the Constraint Correction for all type of data
//////////////////////////////////////////////////////////////////////////

template<class DataTypes>
typename PrecomputedConstraintCorrection<DataTypes>::InverseStorage* PrecomputedConstraintCorrection<DataTypes>::getInverse(std::string name)
{
    std::map< std::string, InverseStorage >& registry = getInverseMap();
    InverseStorage* m = &(registry[name]);
    ++m->nbref;
    return m;
}

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::releaseInverse(std::string name, InverseStorage* inv)
{
    if (inv == NULL) return;
    std::map< std::string, InverseStorage >& registry = getInverseMap();
    if (--inv->nbref == 0)
    {
        if (inv->data) delete[] inv->data;
        registry.erase(name);
    }
}


struct ConstraintActivation { bool acc, vel, pos; };


template<class DataTypes>
std::string PrecomputedConstraintCorrection<DataTypes>::buildFileName()
{
    double dt = this->getContext()->getDt();
    const std::string name = this->getContext()->getName();

    std::stringstream ss;
    ss << name << "-" << nbRows << "-" << dt <<".comp";

    return ss.str();
}



template<class DataTypes>
bool PrecomputedConstraintCorrection<DataTypes>::loadCompliance(std::string fileName)
{
    // Try to load from memory
    sout << "Try to load compliance from memory " << fileName << sendl;

    invM = getInverse(fileName);
    dimensionAppCompliance = nbRows;

    if (invM->data == NULL)
    {
        // Try to load from file
        sout << "Try to load compliance from : " << fileName << sendl;

        if ((sofa::helper::system::DataRepository.findFile(fileName)) && (recompute.getValue() == false))
        {
            invM->data = new Real[nbRows * nbCols];

            std::ifstream compFileIn(fileName.c_str(), std::ifstream::binary);

            sout << "File " << fileName << " found. Loading..." << sendl;

            compFileIn.read((char*)invM->data, nbCols * nbRows * sizeof(double));
            compFileIn.close();

            return true;
        }

        return false;
    }

    return true;
}



template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::saveCompliance(const std::string fileName)
{
    sout << "saveCompliance in " << fileName << sendl;

    std::ofstream compFileOut(fileName.c_str(), std::fstream::out | std::fstream::binary);
    compFileOut.write((char*)invM->data, nbCols * nbRows * sizeof(double));
    compFileOut.close();
}



template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::bwdInit()
{
    f_rotations.beginEdit();
    f_restRotations.beginEdit();
    mstate = dynamic_cast< behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    const VecDeriv& v0 = *mstate->getV();

    nbNodes = v0.size();

    if (nbNodes == 0)
    {
        serr << "No degree of freedom" << sendl;
        return;
    }

    dof_on_node = v0[0].size();

    nbRows = nbNodes * dof_on_node;
    nbCols = nbNodes * dof_on_node;

    double dt = this->getContext()->getDt();

    invName = f_fileCompliance.getFullPath().empty() ? buildFileName() : f_fileCompliance.getFullPath();

    if (!loadCompliance(invName))
    {
        sout << "Compliance being built" << sendl;

        // Buffer Allocation
        invM->data = new Real[nbRows * nbCols];

        // for the intial computation, the gravity has to be put at 0
        const Vec3d gravity = this->getContext()->getGravityInWorld();
        const Vec3d gravity_zero(0.0,0.0,0.0);
        this->getContext()->setGravityInWorld(gravity_zero);

        EulerImplicitSolver* eulerSolver;
        CGLinearSolver< GraphScatteredMatrix, GraphScatteredVector >* cgLinearSolver;
        core::behavior::LinearSolver* linearSolver;

        this->getContext()->get(eulerSolver);
        this->getContext()->get(cgLinearSolver);
        this->getContext()->get(linearSolver);

        simulation::Node *solvernode = NULL;

        if (eulerSolver && cgLinearSolver)
        {
            sout << "use EulerImplicitSolver & CGLinearSolver" << sendl;
            solvernode = (simulation::Node*)eulerSolver->getContext();
        }
        else if (eulerSolver && linearSolver)
        {
            sout << "use EulerImplicitSolver & LinearSolver" << sendl;
            solvernode = (simulation::Node*)eulerSolver->getContext();
        }
        else if(eulerSolver)
        {
            sout << "use EulerImplicitSolver" << sendl;
            solvernode = (simulation::Node*)eulerSolver->getContext();
        }
        else
        {
            serr << "PrecomputedContactCorrection must be associated with EulerImplicitSolver+LinearSolver for the precomputation\nNo Precomputation" << sendl;
            return;
        }

#ifdef SOFA_HAVE_EIGEN2
        helper::vector< sofa::component::constraintset::LMConstraintSolver* > listLMConstraintSolver;
        solvernode->get< sofa::component::constraintset::LMConstraintSolver >(&listLMConstraintSolver, core::objectmodel::BaseContext::SearchDown);
        helper::vector< ConstraintActivation > listConstraintActivation(listLMConstraintSolver.size());
        for (unsigned int i=0; i<listLMConstraintSolver.size(); ++i)
        {
            listConstraintActivation[i].acc=listLMConstraintSolver[i]->constraintAcc.getValue();
            listLMConstraintSolver[i]->constraintAcc.setValue(false);

            listConstraintActivation[i].vel=listLMConstraintSolver[i]->constraintVel.getValue();
            listLMConstraintSolver[i]->constraintVel.setValue(false);

            listConstraintActivation[i].pos=listLMConstraintSolver[i]->constraintPos.getValue();
            listLMConstraintSolver[i]->constraintPos.setValue(false);
        }
#endif

        //complianceLoaded = true;
        VecDeriv& force = *mstate->getExternalForces();
        force.clear();
        force.resize(nbNodes);
        //v.clear();
        //v.resize(v0.size());//computeDf

        ///////////////////////// CHANGE THE PARAMETERS OF THE SOLVER /////////////////////////////////
        double buf_tolerance = 0, buf_threshold = 0;
        int	   buf_maxIter = 0;

        if (cgLinearSolver)
        {
            buf_tolerance = (double) cgLinearSolver->f_tolerance.getValue();
            buf_maxIter   = (int) cgLinearSolver->f_maxIter.getValue();
            buf_threshold = (double) cgLinearSolver->f_smallDenominatorThreshold.getValue();
            cgLinearSolver->f_tolerance.setValue(1e-20);
            cgLinearSolver->f_maxIter.setValue(5000);
            cgLinearSolver->f_smallDenominatorThreshold.setValue(1e-35);
        }
        ///////////////////////////////////////////////////////////////////////////////////////////////

        VecDeriv& velocity = *mstate->getV();
        VecDeriv velocity0 = *mstate->getV();
        VecCoord& pos = *mstate->getX();
        VecCoord  pos0 = *mstate->getX();

        /// christian : it seems necessary to called the integration one time for initialization
        /// (avoid to have a line of 0 at the top of the matrix)
        if (eulerSolver)
        {
            using core::behavior::BaseMechanicalState;
            eulerSolver->solve(dt, BaseMechanicalState::VecId::position(), BaseMechanicalState::VecId::velocity());
        }

        for (unsigned int f = 0; f < nbNodes; f++)
        {
            std::streamsize prevPrecision = std::cout.precision();
            std::cout.precision(2);
            std::cout << "Precomputing constraint correction : " << std::fixed << (float)f/(float)nbNodes*100.0f << " %   " << '\xd';
            std::cout.flush();
            std::cout.precision(prevPrecision);
            //  serr << "inverse cols node : " << f << sendl;
            Deriv unitary_force;

            for (unsigned int i = 0; i < dof_on_node; i++)
            {
                unitary_force.clear();
                //serr<<"dof n:"<<i<<sendl;
                unitary_force[i]=1.0;
                force[f] = unitary_force;
                ////// reset Position and Velocities ///////
                velocity.clear();
                velocity.resize(nbNodes);
                for (unsigned int n=0; n<nbNodes; n++)
                    pos[n] = pos0[n];
                ////////////////////////////////////////////
                //serr<<"pos0 set"<<sendl;

                /*	if (f*dof_on_node+i < 2)
                	{
                		eulerSolver->f_verbose.setValue(true);
                		eulerSolver->f_printLog.setValue(true);
                	//	serr<<"getF : "<<force<<sendl;
                	}*/

                double fact = 1.0; // christian : it is not a compliance... but an admittance that is computed !
                if (eulerSolver)
                    fact = eulerSolver->getPositionIntegrationFactor(); // here, we compute a compliance

                if(eulerSolver)
                {
                    using core::behavior::BaseMechanicalState;
                    eulerSolver->solve(dt, BaseMechanicalState::VecId::position(), BaseMechanicalState::VecId::velocity());
                    if (linearSolver)
                        linearSolver->freezeSystemMatrix(); // do not recompute the matrix for the rest of the precomputation
                }

                //serr<<"solve reussi"<<sendl;

                velocity = *mstate->getV();
                fact /= unitary_force[i];

                /*	if (f*dof_on_node+i < 2)
                	{
                		//eulerSolver->solve(dt, core::behavior::BaseMechanicalState::VecId::position(), core::behavior::BaseMechanicalState::VecId::velocity());
                		eulerSolver->f_verbose.setValue(false);
                		eulerSolver->f_printLog.setValue(false);
                	//	serr<<"getV : "<<velocity<<sendl;
                	}*/

                for (unsigned int v=0; v<nbNodes; v++)
                {
                    for (unsigned int j=0; j<dof_on_node; j++)
                    {
                        invM->data[(v*dof_on_node+j)*nbCols + (f*dof_on_node+i) ] = (Real)(fact * velocity[v][j]);
                    }
                }
                //serr<<"put in appComp"<<sendl;
            }
            unitary_force.clear();
            force[f] = unitary_force;
        }
        if (linearSolver)
            linearSolver->updateSystemMatrix(); // do not recompute the matrix for the rest of the precomputation

        ///////////////////////// RESET PARAMETERS AT THEIR PREVIOUS VALUE /////////////////////////////////
        // gravity is reset at its previous value
        this->getContext()->setGravityInWorld(gravity);
        if (cgLinearSolver)
        {
            cgLinearSolver->f_tolerance.setValue(buf_tolerance);
            cgLinearSolver->f_maxIter.setValue(buf_maxIter);
            cgLinearSolver->f_smallDenominatorThreshold.setValue(buf_threshold);
        }
        ///////////////////////////////////////////////////////////////////////////////////////////////

        saveCompliance(invName);

        //Reset the velocity
        for (unsigned int i=0; i<velocity.size(); i++)
            velocity[i] = velocity0[i];
        //Reset the position
        for (unsigned int i=0; i<pos.size(); i++)
            pos[i] = pos0[i];


#ifdef SOFA_HAVE_EIGEN2
        for (unsigned int i=0; i<listLMConstraintSolver.size(); ++i)
        {
            listLMConstraintSolver[i]->constraintAcc.setValue(listConstraintActivation[i].acc);
            listLMConstraintSolver[i]->constraintVel.setValue(listConstraintActivation[i].vel);
            listLMConstraintSolver[i]->constraintPos.setValue(listConstraintActivation[i].pos);
        }
#endif
    }

    std::cout << "appCompliance = invM->data\n";
    appCompliance = invM->data;

    // Optimisation for the computation of W
    _indexNodeSparseCompliance.resize(v0.size());
    //_sparseCompliance.resize(v0.size()*MAX_NUM_CONSTRAINT_PER_NODE);


    ////  debug print 400 first row and column of the matrix
    //if (this->f_printLog.getValue())
    //{
    //	sout << "Matrix compliance : nbCols = " << nbCols << "  nbRows =" << nbRows;

    //	for (unsigned int i = 0; i < 20 && i < nbCols; i++)
    //	{
    //		sout << sendl;
    //		for (unsigned int j = 0; j < 20 && j < nbCols; j++)
    //		{
    //			sout << " \t " << appCompliance[j*nbCols + i];
    //		}
    //	}

    //	sout << sendl;
    //}
    ////sout << "quit init "  << endl;

    //sout << "----------- Test Quaternions --------------" << sendl;

    //// rotation de -Pi/2 autour de z en init
    //Quat q0(0,0,-0.7071067811865475, 0.7071067811865475);
    //q0.normalize();


    //// rotation de -Pi/2 autour de x dans le repËre dÈfini par q0; (=rotation Pi/2 autour de l'axe y dans le repËre global)
    //Quat q_q0(-0.7071067811865475,0,0,0.7071067811865475);
    //q_q0.normalize();


    //// calcul de la rotation Èquivalente dans le repËre global;
    //Quat q = q0 * q_q0;
    //q.normalize();

    //// test des rotations:
    //sout<<"VecX = "<<q.rotate( Vec3d(1.0,0.0,0.0) )<<sendl;
    //sout<<"VecY = "<<q.rotate( Vec3d(0.0,1.0,0.0) )<<sendl;
    //sout<<"VecZ = "<<q.rotate( Vec3d(0.0,0.0,1.0) )<<sendl;


    //// on veut maintenant retrouver l'Èquivalent de q_q0 dans le repËre global
    //// c'est ‡ dire une rotation de Pi/2 autour de l'axe y
    //Quat q_test = q * q0.inverse();

    //sout<<"q_test = "<<q_test<<sendl;

    //sout<<"Alpha = "<<q_test.toEulerVector()<< " doit valoir une rotation de Pi/2 autour de l'axe y"<<sendl;
}


template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::getCompliance(defaulttype::BaseMatrix* W)
{
    const VecConst& constraints = *mstate->getC();

    unsigned int numConstraints = constraints.size();

    /////////// The constraints on the same nodes are gathered //////////////////////
    //gatherConstraints();
    /////////////////////////////////////////////////////////////////////////////////

    /////////// The constraints are modified using a rotation value at each node/////
    if(_rotations)
        rotateConstraints(false);
    /////////////////////////////////////////////////////////////////////////////////


    /////////// Which node are involved with the contact ?/////
    //std::list<int> activeDof;

    unsigned int noSparseComplianceSize = _indexNodeSparseCompliance.size();

    for (unsigned int i = 0; i < noSparseComplianceSize; ++i)
    {
        _indexNodeSparseCompliance[i] = -1;
    }

    int nActiveDof = 0;

    for(unsigned int c1 = 0; c1 < numConstraints; c1++)
    {
        ConstConstraintIterator itConstraint;
        std::pair< ConstConstraintIterator, ConstConstraintIterator > iter = constraints[c1].data();

        for (itConstraint = iter.first; itConstraint != iter.second; itConstraint++)
        {
            unsigned int dof = itConstraint->first;
            //activeDof.push_back(dof);

            if (_indexNodeSparseCompliance[dof] != 0)
            {
                ++nActiveDof;
                _indexNodeSparseCompliance[dof] = 0;
            }
        }
    }
    //unsigned int numNodes1 = activeDof.size();
    //sout<< "numNodes : avant = "<<numNodes1;
    //activeDof.sort();
    //activeDof.unique();
    //	unsigned int numNodes = activeDof.size();
    //sout<< " apres = "<<numNodes<<sendl;

    // Commented by PJ
    /*
    int nActiveDof = 0;
    for (unsigned int i = 0; i < noSparseComplianceSize; ++i)
    {
    	if (_indexNodeSparseCompliance[i] == 0)
    		++nActiveDof;
    }
    */

    ////////////////////////////////////////////////////////////
    unsigned int offset, offset2;
    unsigned int ii,jj, curRowConst, curColConst, it;
    Deriv Vbuf;
    int indexCurColConst, indexCurRowConst;
    it=0;

    //////////////////////////////////////////
    //std::vector<Deriv> sparseCompliance;
    _sparseCompliance.resize(nActiveDof * numConstraints);

    //std::list<int>::iterator IterateurListe;
    //for(IterateurListe=activeDof.begin();IterateurListe!=activeDof.end();IterateurListe++)
    //  {
    //  int NodeIdx = (*IterateurListe);

    for (int NodeIdx = 0; NodeIdx < (int)noSparseComplianceSize; ++NodeIdx)
    {
        if (_indexNodeSparseCompliance[NodeIdx] == -1)
            continue;

        _indexNodeSparseCompliance[NodeIdx] = it;

        for (curColConst = 0; curColConst < numConstraints; curColConst++)
        {
            indexCurColConst = mstate->getConstraintId()[curColConst];

            Vbuf.clear();
            ConstConstraintIterator itConstraint;

            std::pair< ConstConstraintIterator, ConstConstraintIterator > iter = constraints[curColConst].data();

            for (itConstraint = iter.first; itConstraint != iter.second; itConstraint++)
            {
                unsigned int dof = itConstraint->first;
                const Deriv n2 = itConstraint->second;
                offset = dof_on_node*(NodeIdx * nbCols +  dof);

                for (ii = 0; ii < dof_on_node; ii++)
                {
                    offset2 = offset+ii*nbCols;
                    for (jj = 0; jj < dof_on_node; jj++)
                    {
                        Vbuf[ii] += appCompliance[offset2 + jj] * n2[jj];
                    }
                }
            }
            _sparseCompliance[it]=Vbuf;
            it++;
        }
    }


    for(curRowConst = 0; curRowConst < numConstraints; curRowConst++)
    {
        indexCurRowConst = mstate->getConstraintId()[curRowConst];//global index of constraint

        ConstConstraintIterator itConstraint;

        std::pair< ConstConstraintIterator, ConstConstraintIterator > iter=constraints[curRowConst].data();
        for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
        {
            const int NodeIdx = itConstraint->first;
            const Deriv n1 = itConstraint->second;

            unsigned int temp =(unsigned int) _indexNodeSparseCompliance[NodeIdx];

            for(curColConst = curRowConst; curColConst < numConstraints; curColConst++)
            {
                indexCurColConst = mstate->getConstraintId()[curColConst];
                double w = _sparseCompliance[temp + curColConst]*n1;
                //W[indexCurRowConst][indexCurColConst] += w;
                //sout << "W("<<indexCurRowConst<<","<<indexCurColConst<<") = "<<w<<sendl;
                W->add(indexCurRowConst, indexCurColConst, w);
                if (indexCurRowConst != indexCurColConst)
                    W->add(indexCurColConst, indexCurRowConst, w);
            }
        }
        /*
        //Compliance matrix is symetric ?
        for(unsigned int curColConst = curRowConst+1; curColConst < numConstraints; curColConst++)
        {
        int indexCurColConst = mstate->getConstraintId()[curColConst];
        W[indexCurColConst][indexCurRowConst] = W[indexCurRowConst][indexCurColConst];
        }
        */
    }
}


template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::applyContactForce(const defaulttype::BaseVector *f)
{
    VecDeriv& force = *mstate->getF();
    VecDeriv& dx = *mstate->getDx();
    VecCoord& x = *mstate->getX();
    VecDeriv& v = *mstate->getV();

    const VecDeriv& v_free = *mstate->getVfree();
    const VecCoord& x_free = *mstate->getXfree();
    const VecConst& constraints = *mstate->getC();

    double dt = this->getContext()->getDt();
    unsigned int numConstraints = constraints.size();

    // ici on fait comme avant
    // Euler integration... will be done in the "integrator" as soon as it exists !
    dx.clear();
    dx.resize(v.size());

    force.clear();
    force.resize(x_free.size());

    std::list<int> activeDof;

    //	sout<<"First list:"<<sendl;
    for(unsigned int c1 = 0; c1 < numConstraints; c1++)
    {
        int indexC1 = mstate->getConstraintId()[c1];

        double fC1 = (Real)f->element(indexC1);
        //sout << "fC("<<indexC1<<")="<<fC1<<sendl;

        if (fC1 != 0.0)
        {
            ConstConstraintIterator itConstraint;

            std::pair< ConstConstraintIterator, ConstConstraintIterator > iter = constraints[c1].data();

            for (itConstraint = iter.first; itConstraint != iter.second; itConstraint++)
            {
                unsigned int dof = itConstraint->first;
                const Deriv n = itConstraint->second;
                //on ne fait pas passer les forces du repere courant a celui initial ?
                // <-non, car elles ont deja ete tournees car on utilise une reference dans getCompliance !!!
                Deriv temp = n * fC1;
                force[dof] += temp;
                activeDof.push_back(dof);
            }
        }
    }

    activeDof.sort();
    activeDof.unique();

    //for (unsigned int i=0; i< force.size(); i++)
    //    sout << "f("<<i<<")="<<force[i]<<sendl;

    std::list<int>::iterator IterateurListe;
    unsigned int i;
    unsigned int offset, offset2;
    for (IterateurListe = activeDof.begin(); IterateurListe != activeDof.end(); IterateurListe++)
    {
        int f = (*IterateurListe);

        for (i=0; i< dof_on_node; i++)
        {
            Fbuf[i] = force[f][i];
        }

        for(unsigned int v = 0 ; v < dx.size() ; v++)
        {
            offset =  v * dof_on_node * nbCols + f*dof_on_node;
            for (unsigned int j=0; j< dof_on_node; j++)
            {
                offset2 = offset+ j*nbCols;
                DXbuf=0.0;
                for (i = 0; i < dof_on_node; i++)
                {
                    DXbuf += appCompliance[ offset2 + i ] * Fbuf[i];
                }
                dx[v][j]+=DXbuf;
            }
        }
    }

    force.clear();
    force.resize(x_free.size());

    if(_rotations)
        rotateResponse();

    for (unsigned int i=0; i< dx.size(); i++)
    {
        x[i] = x_free[i];
        v[i] = v_free[i];

        x[i] += dx[i];
        v[i] += dx[i] * (1/dt);
    }
}


template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::getComplianceMatrix(defaulttype::BaseMatrix* m) const
{
    m->resize(dimensionAppCompliance,dimensionAppCompliance);

    for (unsigned int l = 0; l < dimensionAppCompliance; ++l)
    {
        for (unsigned int c = 0; c < dimensionAppCompliance; ++c)
        {
            m->set(l, c, appCompliance[l * dimensionAppCompliance + c]);
        }
    }
}

/*
template<>
void PrecomputedConstraintCorrection<defaulttype::Rigid3Types>::applyContactForce(double *f)
{
	VecDeriv& force = *mstate->getExternalForces();
	const VecConst& constraints = *mstate->getC();
	Deriv weighedNormal;

	const sofa::defaulttype::Rigid3Mass* massValue;

	simulation::tree::GNode *node = dynamic_cast<simulation::tree::GNode *>(getContext());

	if (node != NULL)
	{
		core::behavior::BaseMass*_m = node->mass;
		component::mass::UniformMass<defaulttype::Rigid3Types, defaulttype::Rigid3Mass> *m = dynamic_cast<component::mass::UniformMass<defaulttype::Rigid3Types, defaulttype::Rigid3Mass>*> (_m);
		massValue = &( m->getMass());
	}
	else
	{
		massValue = new sofa::defaulttype::Rigid3Mass();
		printf("\n WARNING : node is not found => massValue could be false in getCompliance function");
	}


	double dt = this->getContext()->getDt();

	force.resize(0);
	force.resize(1);
	force[0] = Deriv();

	int numConstraints = constraints.size();

	for(int c1 = 0; c1 < numConstraints; c1++)
	{
		int indexC1 = mstate->getConstraintId()[c1];

		if (f[indexC1] != 0.0)
		{
			int sizeC1 = constraints[c1].size();
			for(int i = 0; i < sizeC1; i++)
			{
				weighedNormal = constraints[c1][i].data; // weighted normal
				force[0].getVCenter() += weighedNormal.getVCenter() * f[indexC1];
				force[0].getVOrientation() += weighedNormal.getVOrientation() * f[indexC1];
			}
		}
	}


	VecDeriv& dx = *mstate->getDx();
	VecCoord& x = *mstate->getX();
	VecDeriv& v = *mstate->getV();
	VecDeriv& v_free = *mstate->getVfree();
	VecCoord& x_free = *mstate->getXfree();


	//	mstate->setX(x_free);
	//	mstate->setV(v_free);
	x[0]=x_free[0];
	v[0]=v_free[0];

	// Euler integration... will be done in the "integrator" as soon as it exists !
	dx.resize(v.size());
	dx[0] = force[0] / (*massValue);
	dx[0] *= dt;
	v[0] += dx[0];
	dx[0] *= dt;
	x[0] += dx[0];
	//	simulation::tree::MechanicalPropagateAndAddDxVisitor(dx).execute(this->getContext());
}


template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1dTypes>::applyContactForce(double *f)
{
	VecDeriv& force = *mstate->getExternalForces();
	const VecConst& constraints = *mstate->getC();
	unsigned int numConstraints = constraints.size();

	force.resize((*mstate->getX()).size());

	for(unsigned int c1 = 0; c1 < numConstraints; c1++)
	{
		int indexC1 = mstate->getConstraintId()[c1];

		if (f[indexC1] != 0.0)
		{
			int sizeC1 = constraints[c1].size();
			for(int i = 0; i < sizeC1; i++)
			{
				force[constraints[c1][i].index] += constraints[c1][i].data * f[indexC1];
			}
		}
	}

	VecDeriv& dx = *mstate->getDx();
	VecCoord& x = *mstate->getX();
	VecDeriv& v = *mstate->getV();
	VecDeriv& v_free = *mstate->getVfree();
	VecCoord& x_free = *mstate->getXfree();
	double dt = this->getContext()->getDt();


	// Euler integration... will be done in the "integrator" as soon as it exists !
	dx.resize(v.size());

	for (unsigned int i=0; i<dx.size(); i++)
	{
		x[i] = x_free[i];
		v[i] = v_free[i];
		dx[i] = force[i]/10000.0;
		x[i] += dx[i];
		v[i] += dx[i]/dt;
	}
}
*/

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::applyPredictiveConstraintForce(const defaulttype::BaseVector *f)
{
    VecDeriv& force = *mstate->getExternalForces();
//	VecDeriv& force = *mstate->getF();

    const unsigned int numDOFs = mstate->getSize();

    force.clear();
    force.resize(numDOFs);
    for (unsigned int i=0; i< numDOFs; i++)
        force[i] = Deriv();

//    if(this->_rotations){
//
//        this->rotateConstraints(true);
//    }

    const VecConst& constraints = *mstate->getC();
    unsigned int numConstraints = constraints.size();

    std::cout.precision(7);
    for(unsigned int c1 = 0; c1 < numConstraints; c1++)
    {
        int indexC1 = mstate->getConstraintId()[c1];
        double fC1 = f->element(indexC1);
        //sout << "fC("<<indexC1<<")="<<fC1<<sendl;
        if (fC1 != 0.0)
        {
            ConstConstraintIterator itConstraint;
            std::pair< ConstConstraintIterator, ConstConstraintIterator > iter=constraints[c1].data();
            for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
            {
                unsigned int dof = itConstraint->first;
                Deriv n = itConstraint->second;

                //std::cout << "Predictive Force => f("<<itConstraint->first<<") += "<< (itConstraint->second * fC1) << std::endl;
                force[dof] += n * fC1;
            }
        }
    }

    std::cout<< "Predictive External Forces: "<< force<<std::endl;

}


template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::resetContactForce()
{
    VecDeriv& force = *mstate->getF();
    for( unsigned i=0; i<force.size(); ++i )
        force[i] = Deriv();
}

////  DRAW : generic function ////
template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::draw()
{
    if (!getContext()->getShowBehaviorModels()) return;
}

#ifndef SOFA_FLOAT
template<>
void PrecomputedConstraintCorrection<defaulttype::Vec3dTypes>::draw();

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::rotateConstraints(bool back)
{
    VecConst& constraints = *mstate->getC();
    unsigned int numConstraints = constraints.size();

    simulation::Node *node = dynamic_cast<simulation::Node *>(getContext());

    sofa::component::forcefield::TetrahedronFEMForceField<DataTypes>* forceField = NULL;
    sofa::component::container::RotationFinder<DataTypes>* rotationFinder = NULL;

    if (node != NULL)
    {
        //		core::behavior::BaseForceField* _forceField = node->forceField[1];
        forceField = node->get<component::forcefield::TetrahedronFEMForceField<DataTypes> > ();
        if (forceField == NULL)
        {
            rotationFinder = node->get<component::container::RotationFinder<DataTypes> > ();
            if (rotationFinder == NULL)
            {
                sout << "No rotation defined : only defined for TetrahedronFEMForceField and RotationFinder!";
                return;
            }
        }
    }
    else
    {
        sout << "Error getting context in method: PrecomputedConstraintCorrection<defaulttype::Vec3dTypes>::rotateConstraints(false)";
        return;
    }

    //sout << "start rotating normals " << g_timer_elapsed(timer, &micro) << sendl;
    //	int sizemax=0;
    //	int index_const = -1;
    // on fait tourner les normales (en les ramenant dans le "pseudo" repere initial) //
    for(unsigned int curRowConst = 0; curRowConst < numConstraints; curRowConst++)
    {
        ConstraintIterator itConstraint;
        std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[curRowConst].data();

        for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
        {
            unsigned int dof = itConstraint->first;
            Deriv& n = itConstraint->second;
            const int localRowNodeIdx = dof;
            Transformation Ri;
            if (forceField != NULL)
            {
                forceField->getRotation(Ri, localRowNodeIdx);
            }
            else // rotationFinder has been defined
            {
                Ri = rotationFinder->getRotations()[localRowNodeIdx];
            }
            if(!back)
                Ri.transpose();
            // on passe les normales du repere global au repere local
            Deriv n_i = Ri * n;
            n.x() =  n_i.x();
            n.y() =  n_i.y();
            n.z() =  n_i.z();
        }
        /*
        // test pour voir si on peut reduire le nombre de contrainte
        if (sizeCurRowConst > sizemax)
        {
        sizemax = sizeCurRowConst;
        index_const = curRowConst;
        }
        */
    }
}

template<>
void PrecomputedConstraintCorrection<defaulttype::Rigid3dTypes>::rotateConstraints(bool back);

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1dTypes>::rotateConstraints(bool back);

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::rotateResponse()
{
    simulation::Node *node = dynamic_cast<simulation::Node *>(getContext());

    sofa::component::forcefield::TetrahedronFEMForceField<DataTypes>* forceField = NULL;
    sofa::component::container::RotationFinder<DataTypes>* rotationFinder = NULL;

    if (node != NULL)
    {
        //		core::behavior::BaseForceField* _forceField = node->forceField[1];
        forceField = node->get<component::forcefield::TetrahedronFEMForceField<DataTypes> > ();
        if (forceField == NULL)
        {
            rotationFinder = node->get<component::container::RotationFinder<DataTypes> > ();
            if (rotationFinder == NULL)
            {
                sout << "No rotation defined : only defined for TetrahedronFEMForceField and RotationFinder!";
                return;
            }
        }
    }
    else
    {
        sout << "Error getting context in method: PrecomputedConstraintCorrection<defaulttype::Vec3dTypes>::rotateConstraints(false)";
        return;
    }

    VecDeriv& dx = *mstate->getDx();
    for(unsigned int j = 0; j < dx.size(); j++)
    {
        Transformation Rj;
        if (forceField != NULL)
        {
            forceField->getRotation(Rj, j);
        }
        else // rotationFinder has been defined
        {
            Rj = rotationFinder->getRotations()[j];
        }
        // on passe les deplacements du repere local au repere global
        const Deriv& temp = Rj * dx[j];
        dx[j] = temp;
    }
}

template<>
void PrecomputedConstraintCorrection<defaulttype::Rigid3dTypes>::rotateResponse();

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1dTypes>::rotateResponse();

#endif
#ifndef SOFA_DOUBLE

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec3fTypes>::rotateConstraints(bool back);

template<>
void PrecomputedConstraintCorrection<defaulttype::Rigid3fTypes>::rotateConstraints(bool back);

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1fTypes>::rotateConstraints(bool back);

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec3fTypes>::rotateResponse();

template<>
void PrecomputedConstraintCorrection<defaulttype::Rigid3fTypes>::rotateResponse();

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1fTypes>::rotateResponse();

#endif


// new API for non building the constraint system during solving process //

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::resetForUnbuiltResolution(double * f, std::list<int>& /*renumbering*/)
{
    constraint_force = f;
    VecConst& constraints = *mstate->getC();
    localConstraintId = &(mstate->getConstraintId());
    unsigned int numConstraints = constraints.size();

#ifdef NEW_METHOD_UNBUILT
    constraint_D.clear();
    constraint_D.resize(mstate->getSize());

    constraint_F.clear();
    constraint_F.resize(mstate->getSize());

    constraint_dofs.clear();

    bool error_message_not_displayed=true;
#endif

    /////////// The constraints on the same nodes are gathered //////////////////////
    //gatherConstraints();
    /////////////////////////////////////////////////////////////////////////////////

    /////////// The constraints are modified using a rotation value at each node/////
    if(_rotations)
        rotateConstraints(false);
    /////////////////////////////////////////////////////////////////////////////////


    /////////// Which node are involved with the contact ?/////

    for(unsigned int c1 = 0; c1 < numConstraints; c1++)
    {
        ConstConstraintIterator itConstraint;
        std::pair< ConstConstraintIterator, ConstConstraintIterator > iter=constraints[c1].data();
        for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
        {
            unsigned int dof = itConstraint->first;
            constraint_dofs.push_back(dof);
        }
    }

//	unsigned int numNodes1 = constraint_dofs.size();
//	sout<< "numNodes : avant = "<<numNodes1;
    constraint_dofs.sort();
    constraint_dofs.unique();
//	unsigned int numNodes = constraint_dofs.size();
//	sout<< " apres = "<<numNodes<<sendl;
//	sout<< "numConstraints = "<< numConstraints<<sendl;

    id_to_localIndex.clear();

    for(unsigned int i=0; i<numConstraints; ++i)
    {
        unsigned int c = (*localConstraintId)[i];
        if (c >= id_to_localIndex.size()) id_to_localIndex.resize(c+1,-1);
        if (id_to_localIndex[c] != -1)
            serr << "duplicate entry in constraints for id " << c << " : " << id_to_localIndex[c] << " + " << i << sendl;
        id_to_localIndex[c] = i;

#ifdef NEW_METHOD_UNBUILT  // Fill constraint_F => provide the present constraint forces
        double fC = f[c];
        // debug
        //std::cout<<"f["<<indexC<<"] = "<<fC<<std::endl;

        if (fC != 0.0)
        {
            if(error_message_not_displayed)
            {
                serr<<"Initial_guess not supported yet in unbuilt mode with NEW_METHOD_UNBUILT!=> PUT F to 0"<<sendl;
                error_message_not_displayed = false;
            }

            f[c] = 0.0;
            /*
            ConstraintIterator itConstraint;
            std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[i].data();

            for (itConstraint=iter.first;itConstraint!=iter.second;itConstraint++)
            {
            	unsigned int dof = itConstraint->first;
            	Deriv n = itConstraint->second;
            	constraint_F[dof] +=n * fC;

            	// TODO : remplacer pour faire + rapide !!
            //	setConstraintDForce(&fC, (int)c, (int)c, true);

            }
            */
        }
#endif
    }


#ifndef NEW_METHOD_UNBUILT
    ////////////////////////////////////////////////////////////
    unsigned int offset, offset2;
    unsigned int ii,jj, curRowConst, curColConst, it;
    Deriv Vbuf;
    //int indexCurColConst, indexCurRowConst;
    it=0;
    //////////////////////////////////////////

    //std::vector<Deriv> sparseCompliance;
    _sparseCompliance.resize(constraint_dofs.size()*numConstraints);
    std::list<int>::iterator IterateurListe;
    for(IterateurListe=constraint_dofs.begin(); IterateurListe!=constraint_dofs.end(); IterateurListe++)
    {
        int NodeIdx = (*IterateurListe);
        _indexNodeSparseCompliance[NodeIdx]=it;
        for(curColConst = 0; curColConst < numConstraints; curColConst++)
        {
            //indexCurColConst = mstate->getConstraintId()[curColConst];

            Vbuf.clear();
            ConstConstraintIterator itConstraint;
            std::pair< ConstConstraintIterator, ConstConstraintIterator > iter=constraints[curColConst].data();
            for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
            {
                unsigned int dof = itConstraint->first;
                const Deriv n2 = itConstraint->second;
                offset = dof_on_node*(NodeIdx * nbCols +  dof);

                for (ii=0; ii<dof_on_node; ii++)
                {
                    offset2 = offset+ii*nbCols;
                    for (jj=0; jj<dof_on_node; jj++)
                    {
                        Vbuf[ii] += appCompliance[offset2 + jj] * n2[jj];
                    }
                }
            }
            _sparseCompliance[it]=Vbuf;
            it++;
        }
    }

    localW.resize(numConstraints,numConstraints);

    for(curRowConst = 0; curRowConst < numConstraints; curRowConst++)
    {
        //indexCurRowConst = mstate->getConstraintId()[curRowConst];//global index of constraint

        ConstConstraintIterator itConstraint;

        std::pair< ConstConstraintIterator, ConstConstraintIterator > iter=constraints[curRowConst].data();
        for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
        {
            const int NodeIdx = itConstraint->first;
            const Deriv n1 = itConstraint->second;

            unsigned int temp =(unsigned int) _indexNodeSparseCompliance[NodeIdx];
            for(curColConst = curRowConst; curColConst < numConstraints; curColConst++)
            {
                //indexCurColConst = mstate->getConstraintId()[curColConst];
                double w = _sparseCompliance[temp + curColConst]*n1;
                //W[indexCurRowConst][indexCurColConst] += w;
                //sout << "W("<<indexCurRowConst<<","<<indexCurColConst<<") = "<<w<<sendl;
                //W->add(indexCurRowConst, indexCurColConst, w);
                //if (indexCurRowConst != indexCurColConst)
                //  W->add(indexCurColConst, indexCurRowConst, w);
                localW.add(curRowConst, curColConst, w);
                if (curRowConst != curColConst)
                    localW.add(curColConst, curRowConst, w);
            }
        }
        /*
        //Compliance matrix is symetric ?
        for(unsigned int curColConst = curRowConst+1; curColConst < numConstraints; curColConst++)
        {
        	int indexCurColConst = mstate->getConstraintId()[curColConst];
        	W[indexCurColConst][indexCurRowConst] = W[indexCurRowConst][indexCurColConst];
        }
        */
    }
#endif
}

template<class DataTypes>
bool PrecomputedConstraintCorrection<DataTypes>::hasConstraintNumber(int index)
{
    return ((std::size_t)index) < id_to_localIndex.size() && id_to_localIndex[index] >= 0;
}

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::addConstraintDisplacement(double *d, int begin,int end)
{
#ifdef NEW_METHOD_UNBUILT

    const VecConst& constraints = *mstate->getC();

    for (int id_=begin; id_<=end; id_++)
    {
        int c = id_to_localIndex[id_];

        ConstConstraintIterator itConstraint;
        std::pair< ConstConstraintIterator, ConstConstraintIterator > iter=constraints[c].data();
        double dc = d[id_];

        for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
        {
            Deriv n = itConstraint->second;
            dc += n * constraint_D[itConstraint->first];
        }
        d[id_] = dc;
    }

#else
    unsigned int numConstraints = localConstraintId->size();

    for (int id_=begin; id_<=end; id_++)
    {
        int c = id_to_localIndex[id_];

        double dc = d[id_];
        for (unsigned int j=0; j<numConstraints; ++j)
            dc += localW.element(c,j) * constraint_force[(*localConstraintId)[j]];


        d[id_] = dc;
    }
#endif
}

template<class DataTypes>
#ifdef NEW_METHOD_UNBUILT
void PrecomputedConstraintCorrection<DataTypes>::setConstraintDForce(double * df, int begin, int end, bool update)
#else
void PrecomputedConstraintCorrection<DataTypes>::setConstraintDForce(double * /*df*/, int /*begin*/, int /*end*/, bool /*update*/)
#endif
{

#ifdef NEW_METHOD_UNBUILT

    /// set a force difference on a set of constraints (between constraint number "begin" and constraint number "end"
    /// if update is false, do nothing
    /// if update is true, it computes the displacements due to this delta of force.
    /// As the contact are uncoupled, a displacement is obtained only on dof involved with the constraints

    const VecConst& constraints = *mstate->getC();

    if (!update)
        return;
    // debug
    //if (end<6)
    //    std::cout<<"addDf - df["<<begin<<" to "<<end<<"] ="<< df[begin] << " " << df[begin+1] << " "<< df[begin+2] << std::endl;




    //for (unsigned int i=0; i< force.size(); i++)
    //    sout << "f("<<i<<")="<<force[i]<<sendl;

    std::list<int>::iterator IterateurListe;
    unsigned int i;
    unsigned int offset, offset2;

    for ( int id_=begin; id_<=end; id_++)
    {
        int c = id_to_localIndex[id_];

        ConstConstraintIterator itConstraint;
        std::pair< ConstConstraintIterator, ConstConstraintIterator > iter=constraints[c].data();
        for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
        {
            Deriv n = itConstraint->second;
            unsigned int dof = itConstraint->first;

            constraint_F[dof] += n * df[id_];

            for (i=0; i< dof_on_node; i++)
            {
                Fbuf[i] = n[i] * df[id_];
            }

            for(IterateurListe=constraint_dofs.begin(); IterateurListe!=constraint_dofs.end(); IterateurListe++)
            {
                int dof2 = (*IterateurListe);
                offset = dof2 * dof_on_node * nbCols + dof*dof_on_node;
                for (unsigned int j=0; j< dof_on_node; j++)
                {
                    offset2 =offset+ j*nbCols;
                    DXbuf=0.0;
                    for (i=0; i< dof_on_node; i++)
                    {
                        DXbuf += appCompliance[ offset2 + i ] * Fbuf[i];
                    }
                    constraint_D[dof2][j]+=DXbuf;


                }
            }

        }
    }








#endif
}

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::getBlockDiagonalCompliance(defaulttype::BaseMatrix* W, int begin, int end)

{
#ifdef NEW_METHOD_UNBUILT

    VecConst& constraints = *mstate->getC();

    int numLocalConstraints = 0;



    std::list<int> localActiveDof;
    std::list<int>::iterator IterateurListe;
    std::vector<int> constraintLocalID;




    for ( int id_=begin; id_<=end; id_++)
    {
        numLocalConstraints++;
        int c = id_to_localIndex[id_];
        constraintLocalID.push_back(id_);

        ConstConstraintIterator itConstraint;
        std::pair< ConstConstraintIterator, ConstConstraintIterator > iter=constraints[c].data();
        for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
        {

            unsigned int dof = itConstraint->first;
            localActiveDof.push_back(dof);
        }
    }


    localActiveDof.sort();
    localActiveDof.unique();



    ////////////////////////////////////////////////////////////
    unsigned int offset, offset2;
    unsigned int ii,jj;
    Deriv Vbuf;
    int indexColInMatrixW, indexRowInMatrixW;
    int it=0;
    int it_localActiveDof=0;

    _sparseCompliance.resize(localActiveDof.size()*numLocalConstraints);








    for(IterateurListe=localActiveDof.begin(); IterateurListe!=localActiveDof.end(); IterateurListe++)
    {
        int dof1 = (*IterateurListe);
        _indexNodeSparseCompliance[dof1] = it_localActiveDof;
        //_indexNodeSparseCompliance.push_back(dof1);
        it_localActiveDof++;

        for ( int id_=begin; id_<=end; id_++)
        {

            int c = id_to_localIndex[id_];

            Vbuf.clear();  // displacement obtained on the active node  dof 1  when apply contact force 1 on constraint c

            ConstConstraintIterator itConstraint;
            std::pair< ConstConstraintIterator, ConstConstraintIterator > iter=constraints[c].data();
            for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
            {
                unsigned int dof2 = itConstraint->first;
                const Deriv n2 = itConstraint->second;

                offset = dof_on_node*(dof1 * nbCols +  dof2);

                for (ii=0; ii<dof_on_node; ii++)
                {
                    offset2 = offset+ii*nbCols;
                    for (jj=0; jj<dof_on_node; jj++)
                    {
                        Vbuf[ii] += appCompliance[offset2 + jj] * n2[jj];
                    }
                }

            }
            //
            _sparseCompliance[it]=Vbuf;   // [it = numLocalConstraints *
            it++;
        }

    }
//////////////


    ConstConstraintIterator itConstraint;
    it=0;





    for ( int id_=begin; id_<=end; id_++)
    {

        int c1 = id_to_localIndex[id_];
        indexRowInMatrixW = mstate->getConstraintId()[c1];//global index of constraint



        std::pair< ConstConstraintIterator, ConstConstraintIterator > iter=constraints[c1].data();
        for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
        {
            const int NodeIdx = itConstraint->first;
            const Deriv n1 = itConstraint->second;

            unsigned int c1_loc =(unsigned int) _indexNodeSparseCompliance[NodeIdx];


            for(int c2_loc = it; c2_loc < numLocalConstraints; c2_loc++)
            {
                int id2 = constraintLocalID[c2_loc];
                int c2 = id_to_localIndex[id2];


                indexColInMatrixW = mstate->getConstraintId()[c2];


                double w = n1* _sparseCompliance[c1_loc * numLocalConstraints + c2_loc];





                W->add(id_, id2, w);
                if (id_ != id2)
                {
                    W->add(id2, id_, w);

                }
            }
        }
        it++;
    }



#else



    for (int id1=begin; id1<=end; id1++)
    {
        int c1 = id_to_localIndex[id1];
        for (int id2= id1; id2<=end; id2++)
        {
            int c2 = id_to_localIndex[id2];
            Real w = localW.element(c1,c2);

            W->add(id1, id2, w);
            if (id1 != id2)
                W->add(id2, id1, w);
        }
    }
#endif

}

/////////////////////////////////////////////////////////////////////////////////


#ifndef SOFA_FLOAT

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec3dTypes>::draw();

template<>
void PrecomputedConstraintCorrection<defaulttype::Rigid3dTypes>::rotateConstraints(bool back);

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1dTypes>::rotateConstraints(bool back);

template<>
void PrecomputedConstraintCorrection<defaulttype::Rigid3dTypes>::rotateResponse();

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1dTypes>::rotateResponse();

#endif

#ifndef SOFA_DOUBLE

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec3fTypes>::rotateConstraints(bool back);

template<>
void PrecomputedConstraintCorrection<defaulttype::Rigid3fTypes>::rotateConstraints(bool back);

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1fTypes>::rotateConstraints(bool back);

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec3fTypes>::rotateResponse();

template<>
void PrecomputedConstraintCorrection<defaulttype::Rigid3fTypes>::rotateResponse();

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1fTypes>::rotateResponse();

#endif

} // namespace collision

} // namespace component

} // namespace sofa

#endif
