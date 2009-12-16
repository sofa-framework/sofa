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
#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_CONTACTCORRECTION_INL
#define SOFA_CORE_COMPONENTMODEL_COLLISION_CONTACTCORRECTION_INL

#include "PrecomputedConstraintCorrection.h"
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/MechanicalVisitor.h>

//compliance computation include
#include <sofa/component/odesolver/CGImplicitSolver.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>

#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>

#include <sofa/component/container/RotationFinder.h>

#include <sofa/helper/gl/DrawManager.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/Quater.h>

//#include <glib.h>
#include <sstream>
#include <list>

namespace sofa
{

namespace component
{

namespace constraint
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
    , filePrefix(initData(&filePrefix, "filePrefix","if not empty, the prefix used for the file containing the compliance matrix"))
    , mstate(mm)
    , invM(NULL)
    , appCompliance(NULL)
{
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
    std::map<std::string, InverseStorage>& registry = getInverseMap();
    InverseStorage* m = &(registry[name]);
    ++m->nbref;
    return m;
}

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::releaseInverse(std::string name, InverseStorage* inv)
{
    if (inv == NULL) return;
    std::map<std::string, InverseStorage>& registry = getInverseMap();
    if (--inv->nbref == 0)
    {
        if (inv->data) delete[] inv->data;
        registry.erase(name);
    }
}


template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::bwdInit()
{
    f_rotations.beginEdit();
    f_restRotations.beginEdit();
    mstate = dynamic_cast< behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    const VecDeriv& v0 = *mstate->getV();

    nbNodes = v0.size();

    if (nbNodes==0)
    {
        serr<<"No degree of freedom" << sendl;
        return;
    }
    dof_on_node = v0[0].size();


    nbRows = nbNodes*dof_on_node;
    nbCols = nbNodes*dof_on_node;
    sout << "size : " << nbRows << " " << nbCols << sendl;
    //appCompliance = new Real[nbRows * nbCols];


    double dt = this->getContext()->getDt();

    if (!filePrefix.getValue().empty())
        invName = filePrefix.getValue();
    else
    {
        std::stringstream ss;
        ss << this->getContext()->getName() << "-" << nbRows << "-" << dt <<".comp";
        invName = ss.str();
    }
    invM = getInverse(invName);
    dimensionAppCompliance=nbRows;

    if (invM->data == NULL)
    {
        invM->data = new Real[nbRows * nbCols];

        sout << "try to open : " << invName << endl;
        if (sofa::helper::system::DataRepository.findFile(invName) && recompute.getValue()==false)
        {
            invName=sofa::helper::system::DataRepository.getFile(invName);
            std::ifstream compFileIn(invName.c_str(), std::ifstream::binary);
            sout << "file open : " << invName << " compliance being loaded" << endl;
            //complianceLoaded = true;
            compFileIn.read((char*)invM->data, nbCols * nbRows*sizeof(double));
            compFileIn.close();
        }
        else
        {
            sout << " compliance being built" << sendl;

            // for the intial computation, the gravity has to be put at 0
            const Vec3d gravity = this->getContext()->getGravityInWorld();
            const Vec3d gravity_zero(0.0,0.0,0.0);
            this->getContext()->setGravityInWorld(gravity_zero);

            CGImplicitSolver* odeSolver;
            EulerImplicitSolver* EulerSolver;
            CGLinearSolver<GraphScatteredMatrix,GraphScatteredVector>* CGlinearSolver;
            core::componentmodel::behavior::LinearSolver* linearSolver;

            this->getContext()->get(odeSolver);
            this->getContext()->get(EulerSolver);
            this->getContext()->get(CGlinearSolver);
            this->getContext()->get(linearSolver);

            if(odeSolver)
                sout << "use CGImplicitSolver " << sendl;
            else if(EulerSolver && CGlinearSolver)
                sout << "use EulerImplicitSolver &  CGLinearSolver" << sendl;
            else if(EulerSolver && linearSolver)
                sout << "use EulerImplicitSolver &  LinearSolver" << sendl;
            else if(EulerSolver)
            {
                sout << "use EulerImplicitSolver" << sendl;
            }
            else
            {
                serr<<"PrecomputedContactCorrection must be associated with CGImplicitSolver or EulerImplicitSolver+LinearSolver for the precomputation\nNo Precomputation" << sendl;
                return;
            }



            //complianceLoaded = true;
            VecDeriv& force = *mstate->getExternalForces();
            force.clear();
            force.resize(nbNodes);
            //v.clear();
            //v.resize(v0.size());//computeDf


            ///////////////////////// CHANGE THE PARAMETERS OF THE SOLVER /////////////////////////////////
            double buf_tolerance=0, buf_threshold=0;
            int	   buf_maxIter=0;
            if(odeSolver)
            {
                buf_tolerance = (double) odeSolver->f_tolerance.getValue();
                buf_maxIter   = (int) odeSolver->f_maxIter.getValue();
                buf_threshold = (double) odeSolver->f_smallDenominatorThreshold.getValue();
                odeSolver->f_tolerance.setValue(1e-20);
                odeSolver->f_maxIter.setValue(5000);
                odeSolver->f_smallDenominatorThreshold.setValue(1e-35);
            }
            else if(CGlinearSolver)
            {
                buf_tolerance = (double) CGlinearSolver->f_tolerance.getValue();
                buf_maxIter   = (int) CGlinearSolver->f_maxIter.getValue();
                buf_threshold = (double) CGlinearSolver->f_smallDenominatorThreshold.getValue();
                CGlinearSolver->f_tolerance.setValue(1e-20);
                CGlinearSolver->f_maxIter.setValue(5000);
                CGlinearSolver->f_smallDenominatorThreshold.setValue(1e-35);
            }
            ///////////////////////////////////////////////////////////////////////////////////////////////

            VecDeriv& velocity = *mstate->getV();
            VecDeriv velocity0 = *mstate->getV();
            VecCoord& pos=*mstate->getX();
            VecCoord  pos0=*mstate->getX();


            /// christian : it seems necessary to called the integration one time for initialization
            /// (avoid to have a line of 0 at the top of the matrix)
            if(EulerSolver)
            {
                //serr<<"EulerSolver"<<sendl;
                EulerSolver->solve(dt, core::componentmodel::behavior::BaseMechanicalState::VecId::position(), core::componentmodel::behavior::BaseMechanicalState::VecId::velocity());
            }
            for(unsigned int f = 0 ; f < nbNodes ; f++)
            {
                std::cout.precision(2);
                std::cout << "Precomputing constraint correction : " << std::fixed << (float)f/(float)nbNodes*100.0f << " %   " << '\xd';
                std::cout.flush();
                //  serr << "inverse cols node : " << f << sendl;
                Deriv unitary_force;

                for (unsigned int i=0; i<dof_on_node; i++)
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

                    if(f*dof_on_node+i <2 )
                    {
                        EulerSolver->f_verbose.setValue(true);
                        EulerSolver->f_printLog.setValue(true);
                        serr<<"getF : "<<force<<sendl;
                    }

                    double fact = 1.0; // christian : it is not a compliance... but an admittance that is computed !
                    if (EulerSolver)
                        fact = EulerSolver->getPositionIntegrationFactor(); // here, we compute a compliance

                    //odeSolver->computeContactForce(force);

                    if(odeSolver)
                    {
                        //serr<<"odeSolver"<<sendl;
                        odeSolver->solve(dt);
                    }
                    else if(EulerSolver)
                    {
                        //serr<<"EulerSolver"<<sendl;
                        EulerSolver->solve(dt, core::componentmodel::behavior::BaseMechanicalState::VecId::position(), core::componentmodel::behavior::BaseMechanicalState::VecId::velocity());
                        if (linearSolver)
                            linearSolver->freezeSystemMatrix(); // do not recompute the matrix for the rest of the precomputation
                    }

                    //serr<<"solve reussi"<<sendl;

                    velocity = *mstate->getV();
                    fact /= unitary_force[i];

                    if(f*dof_on_node+i < 2)
                    {

                        //EulerSolver->solve(dt, core::componentmodel::behavior::BaseMechanicalState::VecId::position(), core::componentmodel::behavior::BaseMechanicalState::VecId::velocity());
                        EulerSolver->f_verbose.setValue(false);
                        EulerSolver->f_printLog.setValue(false);
                        serr<<"getV : "<<velocity<<sendl;
                    }
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
            if(odeSolver)
            {
                odeSolver->f_tolerance.setValue(buf_tolerance);
                odeSolver->f_maxIter.setValue(buf_maxIter);
                odeSolver->f_smallDenominatorThreshold.setValue(buf_threshold);
            }
            else if(CGlinearSolver)
            {
                CGlinearSolver->f_tolerance.setValue(buf_tolerance);
                CGlinearSolver->f_maxIter.setValue(buf_maxIter);
                CGlinearSolver->f_smallDenominatorThreshold.setValue(buf_threshold);
            }
            ///////////////////////////////////////////////////////////////////////////////////////////////
            std::ofstream compFileOut(invName.c_str(), std::fstream::out | std::fstream::binary);
            compFileOut.write((char*)invM->data, nbCols * nbRows*sizeof(double));
            compFileOut.close();

            //Reset the velocity
            for (unsigned int i=0; i<velocity.size(); i++) velocity[i]=velocity0[i];
            //Reset the position
            for (unsigned int i=0; i<pos.size(); i++)      pos[i]=pos0[i];
        }
    }

    appCompliance = invM->data;

    // Optimisation for the computation of W
    _indexNodeSparseCompliance.resize(v0.size());
    //_sparseCompliance.resize(v0.size()*MAX_NUM_CONSTRAINT_PER_NODE);


    ////  debug print 400 first row and column of the matrix
    if (this->f_printLog.getValue())
    {
        sout << "Matrix compliance : nbCols ="<<nbCols<<"  nbRows ="<<nbRows ;

        for (unsigned int i=0; i<20 && i<nbCols; i++)
        {
            sout << sendl;
            for (unsigned int j=0; j<20 && j<nbCols; j++)
            {
                sout <<" \t "<< appCompliance[j*nbCols + i];
            }
        }

        sout << sendl;
    }
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

    VecConst& constraints = *mstate->getC();

    unsigned int numConstraints = constraints.size();

    /////////// The constraints on the same nodes are gathered //////////////////////
    //gatherConstraints();
    /////////////////////////////////////////////////////////////////////////////////

    /////////// The constraints are modified using a rotation value at each node/////
    if(_rotations)
        rotateConstraints();
    /////////////////////////////////////////////////////////////////////////////////


    /////////// Which node are involved with the contact ?/////
    //std::list<int> activeDof;
    for (unsigned int i=0; i<_indexNodeSparseCompliance.size(); ++i)
        _indexNodeSparseCompliance[i] = -1;
    for(unsigned int c1 = 0; c1 < numConstraints; c1++)
    {
        ConstConstraintIterator itConstraint;
        std::pair< ConstConstraintIterator, ConstConstraintIterator > iter=constraints[c1].data();

        for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
        {
            unsigned int dof = itConstraint->first;
            //activeDof.push_back(dof);
            _indexNodeSparseCompliance[dof]=0;
        }
    }
    //unsigned int numNodes1 = activeDof.size();
    //sout<< "numNodes : avant = "<<numNodes1;
    //activeDof.sort();
    //activeDof.unique();
    //	unsigned int numNodes = activeDof.size();
    //sout<< " apres = "<<numNodes<<sendl;
    int nActiveDof = 0;
    for (unsigned int i=0; i<_indexNodeSparseCompliance.size(); ++i)
        if (_indexNodeSparseCompliance[i]==0) ++nActiveDof;

    ////////////////////////////////////////////////////////////
    unsigned int offset, offset2;
    unsigned int ii,jj, curRowConst, curColConst, it;
    Deriv Vbuf;
    int indexCurColConst, indexCurRowConst;
    it=0;

    //////////////////////////////////////////
    //std::vector<Deriv> sparseCompliance;
    _sparseCompliance.resize(nActiveDof*numConstraints);
    //std::list<int>::iterator IterateurListe;
    //for(IterateurListe=activeDof.begin();IterateurListe!=activeDof.end();IterateurListe++)
    //  {
    //  int NodeIdx = (*IterateurListe);
    for (int NodeIdx = 0; NodeIdx < (int)_indexNodeSparseCompliance.size(); ++NodeIdx)
    {
        if (_indexNodeSparseCompliance[NodeIdx] == -1) continue;
        _indexNodeSparseCompliance[NodeIdx]=it;
        for(curColConst = 0; curColConst < numConstraints; curColConst++)
        {
            indexCurColConst = mstate->getConstraintId()[curColConst];

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
    VecDeriv& force = *mstate->getExternalForces();
    VecConst& constraints = *mstate->getC();
    unsigned int numConstraints = constraints.size();

    VecDeriv& dx = *mstate->getDx();
    VecCoord& x = *mstate->getX();
    VecDeriv& v = *mstate->getV();
    VecDeriv v_free = *mstate->getVfree();
    VecCoord x_free = *mstate->getXfree();
    double dt = this->getContext()->getDt();


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

            std::pair< ConstConstraintIterator, ConstConstraintIterator > iter=constraints[c1].data();
            for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
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
    for(IterateurListe=activeDof.begin(); IterateurListe!=activeDof.end(); IterateurListe++)
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
                for (i=0; i< dof_on_node; i++)
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
        //sout << "dx("<<i<<")="<<dx[i]<<sendl;
        x[i] = x_free[i];
        v[i] = v_free[i];

        x[i] += dx[i];
        v[i] += dx[i]*(1/dt);
    }
}



template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::getComplianceMatrix(defaulttype::BaseMatrix* m) const
{
    m->resize(dimensionAppCompliance,dimensionAppCompliance);
    for (unsigned int l=0; l<dimensionAppCompliance; ++l)
    {
        for (unsigned int c=0; c<dimensionAppCompliance; ++c)
        {
            m->set(l,c,appCompliance[l*dimensionAppCompliance+c]);
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
		core::componentmodel::behavior::BaseMass*_m = node->mass;
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
void PrecomputedConstraintCorrection<DataTypes>::resetContactForce()
{
    VecDeriv& force = *mstate->getExternalForces();
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
void PrecomputedConstraintCorrection<DataTypes>::rotateConstraints()
{
    VecConst& constraints = *mstate->getC();
    unsigned int numConstraints = constraints.size();

    simulation::Node *node = dynamic_cast<simulation::Node *>(getContext());

    sofa::component::forcefield::TetrahedronFEMForceField<DataTypes>* forceField = NULL;
    sofa::component::container::RotationFinder<DataTypes>* rotationFinder = NULL;

    if (node != NULL)
    {
        //		core::componentmodel::behavior::BaseForceField* _forceField = node->forceField[1];
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
        sout << "Error getting context in method: PrecomputedConstraintCorrection<defaulttype::Vec3dTypes>::rotateConstraints()";
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
void PrecomputedConstraintCorrection<defaulttype::Rigid3dTypes>::rotateConstraints();

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1dTypes>::rotateConstraints();

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::rotateResponse()
{
    simulation::Node *node = dynamic_cast<simulation::Node *>(getContext());

    sofa::component::forcefield::TetrahedronFEMForceField<DataTypes>* forceField = NULL;
    sofa::component::container::RotationFinder<DataTypes>* rotationFinder = NULL;

    if (node != NULL)
    {
        //		core::componentmodel::behavior::BaseForceField* _forceField = node->forceField[1];
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
        sout << "Error getting context in method: PrecomputedConstraintCorrection<defaulttype::Vec3dTypes>::rotateConstraints()";
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
void PrecomputedConstraintCorrection<defaulttype::Vec3fTypes>::rotateConstraints();

template<>
void PrecomputedConstraintCorrection<defaulttype::Rigid3fTypes>::rotateConstraints();

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1fTypes>::rotateConstraints();

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

    /////////// The constraints on the same nodes are gathered //////////////////////
    //gatherConstraints();
    /////////////////////////////////////////////////////////////////////////////////

    /////////// The constraints are modified using a rotation value at each node/////
    if(_rotations)
        rotateConstraints();
    /////////////////////////////////////////////////////////////////////////////////


    /////////// Which node are involved with the contact ?/////
    std::list<int> activeDof;
    for(unsigned int c1 = 0; c1 < numConstraints; c1++)
    {
        ConstConstraintIterator itConstraint;
        std::pair< ConstConstraintIterator, ConstConstraintIterator > iter=constraints[c1].data();
        for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
        {
            unsigned int dof = itConstraint->first;
            activeDof.push_back(dof);
        }
    }
    //unsigned int numNodes1 = activeDof.size();
    //sout<< "numNodes : avant = "<<numNodes1;
    activeDof.sort();
    activeDof.unique();
    //	unsigned int numNodes = activeDof.size();
    //sout<< " apres = "<<numNodes<<sendl;


    ////////////////////////////////////////////////////////////
    unsigned int offset, offset2;
    unsigned int ii,jj, curRowConst, curColConst, it;
    Deriv Vbuf;
    //int indexCurColConst, indexCurRowConst;
    it=0;

    //////////////////////////////////////////
    //std::vector<Deriv> sparseCompliance;
    _sparseCompliance.resize(activeDof.size()*numConstraints);
    std::list<int>::iterator IterateurListe;
    for(IterateurListe=activeDof.begin(); IterateurListe!=activeDof.end(); IterateurListe++)
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

    id_to_localIndex.clear();
    for(unsigned int i=0; i<numConstraints; ++i)
    {
        unsigned int c = (*localConstraintId)[i];
        if (c >= id_to_localIndex.size()) id_to_localIndex.resize(c+1,-1);
        if (id_to_localIndex[c] != -1)
            serr << "duplicate entry in constraints for id " << c << " : " << id_to_localIndex[c] << " + " << i << sendl;
        id_to_localIndex[c] = i;
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
}

template<class DataTypes>
bool PrecomputedConstraintCorrection<DataTypes>::hasConstraintNumber(int index)
{
    return ((std::size_t)index) < id_to_localIndex.size() && id_to_localIndex[index] >= 0;
}

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::addConstraintDisplacement(double *d, int begin,int end)
{
    unsigned int numConstraints = localConstraintId->size();
    for (int id_=begin; id_<=end; id_++)
    {
        int c = id_to_localIndex[id_];
        double dc = d[id_];
        for (unsigned int j=0; j<numConstraints; ++j)
            dc += localW.element(c,j) * constraint_force[(*localConstraintId)[j]];
        d[id_] = dc;
    }
}

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::setConstraintDForce(double * /*df*/, int /*begin*/, int /*end*/, bool /*update*/)
{
}

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::getBlockDiagonalCompliance(defaulttype::BaseMatrix* W, int begin, int end)
{
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
}

/////////////////////////////////////////////////////////////////////////////////

} // namespace collision

} // namespace component

} // namespace sofa

#endif
