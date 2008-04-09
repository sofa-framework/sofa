/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_CONTACTCORRECTION_INL
#define SOFA_CORE_COMPONENTMODEL_COLLISION_CONTACTCORRECTION_INL

#include "PrecomputedConstraintCorrection.h"
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/component/mass/UniformMass.h>
#include <sofa/simulation/tree/MechanicalVisitor.h>

//compliance computation include
#include <sofa/component/odesolver/CGImplicitSolver.h>
//#include <glib.h>
#include <sstream>
#include <list>

namespace sofa
{

namespace component
{

namespace constraint
{
#define	MAX_NUM_CONSTRAINT_PER_NODE 100
#define EPS_UNITARY_FORCE 0.01

using namespace sofa::component::odesolver;

template<class DataTypes>
PrecomputedConstraintCorrection<DataTypes>::PrecomputedConstraintCorrection(behavior::MechanicalState<DataTypes> *mm)
    : _rotations(false)
    , f_rotations(initDataPtr(&f_rotations,&_rotations,"rotations",""))
    , _restRotations(false)
    , f_restRotations(initDataPtr(&f_restRotations,&_restRotations,"restDeformations",""))
    , mstate(mm)
    , appCompliance(NULL)
    , _indexNodeSparseCompliance(NULL)
{
}

template<class DataTypes>
PrecomputedConstraintCorrection<DataTypes>::~PrecomputedConstraintCorrection()
{
    if(appCompliance != NULL)
        delete [] appCompliance;
    if(_indexNodeSparseCompliance != NULL)
        delete [] _indexNodeSparseCompliance;
}



//////////////////////////////////////////////////////////////////////////
//   Precomputation of the Constraint Correction for all type of data
//////////////////////////////////////////////////////////////////////////

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::init()
{
    mstate = dynamic_cast< behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    const VecDeriv& v0 = *mstate->getV();

    nbNodes = v0.size();

    if (nbNodes==0)
    {
        std::cout << "WARNING : No degree of freedom" << endl;;
        return;
    }
    dof_on_node = v0[0].size();


    nbRows = nbNodes*dof_on_node;
    nbCols = nbNodes*dof_on_node;
    std::cout << "size : " << nbRows << " " << nbCols << std::endl;
    appCompliance = new double[nbRows * nbCols];


    double dt = this->getContext()->getDt();

    std::stringstream ss;

    ss << this->getContext()->getName() << ".comp";

    std::ifstream compFileIn(ss.str().c_str(), std::ifstream::binary);

    std::cout << "try to open : " << ss.str() << endl;

    if(compFileIn.good())
    {
        std::cout << "file open : " << ss.str() << " compliance being loaded" << endl;
        //complianceLoaded = true;
        compFileIn.read((char*)appCompliance, nbCols * nbRows*sizeof(double));
        compFileIn.close();
    }
    else
    {
        std::cout << "can not open : " << ss.str() << " compliance being built" << endl;

        // for the intial computation, the gravity has to be put at 0
        const Vec3d gravity = this->getContext()->getGravityInWorld();
        const Vec3d gravity_zero(0.0,0.0,0.0);
        this->getContext()->setGravityInWorld(gravity_zero);

        //complianceLoaded = true;
        VecDeriv& force = *mstate->getExternalForces();
        force.clear();
        force.resize(nbNodes);
        //v.clear();
        //v.resize(v0.size());//computeDf
        CGImplicitSolver* odeSolver = dynamic_cast<CGImplicitSolver*>(dynamic_cast<simulation::tree::GNode *>(this->getContext())->solver[0]);

        if (odeSolver==NULL)
        {
            std::cout << "WARNING : PrecomputedContactCorrection Must be associated with CGImplicitSolver for the precomputation " << std::endl;
            std::cout << "No Precomputation " << std::endl;
            return;
        }
        else
            std::cout << "use solver CGImplicitSolver " << std::endl;

        ///////////////////////// CHANGE THE PARAMETERS OF THE SOLVER /////////////////////////////////
        double buf_tolerance = (double) odeSolver->f_tolerance.getValue();
        int	   buf_maxIter   = (int) odeSolver->f_maxIter.getValue();
        double buf_threshold = (double) odeSolver->f_smallDenominatorThreshold.getValue();
        odeSolver->f_tolerance.setValue(1e-20);
        odeSolver->f_maxIter.setValue(500);
        odeSolver->f_smallDenominatorThreshold.setValue(1e-35);
        ///////////////////////////////////////////////////////////////////////////////////////////////

        VecDeriv& velocity = *mstate->getV();
        VecCoord& pos=*mstate->getX();
        VecCoord& pos0=*mstate->getX0();



        for(unsigned int f = 0 ; f < nbNodes ; f++)
        {
            std::cout << "inverse cols node : " << f << std::endl;
            Deriv unitary_force;

            for (unsigned int i=0; i<dof_on_node; i++)
            {
                unitary_force.clear();
                unitary_force[i]=1.0;
                force[f] = unitary_force;
                ////// reset Position and Velocities ///////
                velocity.clear();
                velocity.resize(nbNodes);
                for (unsigned int n=0; n<nbNodes; n++)
                    pos[n] = pos0[n];
                ////////////////////////////////////////////


                //odeSolver->computeContactForce(force);
                odeSolver->solve(dt);
                velocity = *mstate->getV();

                for (unsigned int v=0; v<nbNodes; v++)
                {

                    for (unsigned int j=0; j<dof_on_node; j++)
                    {
                        appCompliance[(v*dof_on_node+j)*nbCols + (f*dof_on_node+i) ] = velocity[v][j] / unitary_force[i];
                    }
                }
            }
            unitary_force.clear();
            force[f] = unitary_force;
        }
        ///////////////////////// RESET PARAMETERS AT THEIR PREVIOUS VALUE /////////////////////////////////
        // gravity is reset at its previous value
        this->getContext()->setGravityInWorld(gravity);
        odeSolver->f_tolerance.setValue(buf_tolerance);
        odeSolver->f_maxIter.setValue(buf_maxIter);
        odeSolver->f_smallDenominatorThreshold.setValue(buf_threshold);
        ///////////////////////////////////////////////////////////////////////////////////////////////
        std::ofstream compFileOut(ss.str().c_str(), std::fstream::out | std::fstream::binary);
        compFileOut.write((char*)appCompliance, nbCols * nbRows*sizeof(double));
        compFileOut.close();
    }

    // Optimisation for the computation of W
    _indexNodeSparseCompliance = new int[v0.size()];
    _sparseCompliance.resize(v0.size()*MAX_NUM_CONSTRAINT_PER_NODE);


    ////  debug print 100 first row and column of the matrix
    //std::cout << "Matrix compliance" ;

    //for (unsigned int i=0; i<100 && i<nbCols; i++){
    //	std::cout << std::endl;
    //	for (unsigned int j=0; j<100 && j<nbCols; j++)
    //	{
    //		std::cout <<" \t "<< appCompliance[j*nbCols + i];
    //	}
    //}
    //std::cout << "quit init "  << endl;

    std::cout << "----------- Test Quaternions --------------" << std::endl;

    // rotation de -Pi/2 autour de z en init
    Quat q0(0,0,-0.7071067811865475, 0.7071067811865475);
    q0.normalize();


    // rotation de -Pi/2 autour de x dans le repËre dÈfini par q0; (=rotation Pi/2 autour de l'axe y dans le repËre global)
    Quat q_q0(-0.7071067811865475,0,0,0.7071067811865475);
    q_q0.normalize();


    // calcul de la rotation Èquivalente dans le repËre global;
    Quat q = q0 * q_q0;
    q.normalize();

    // test des rotations:
    std::cout<<"VecX = "<<q.rotate( Vec3d(1.0,0.0,0.0) )<<std::endl;
    std::cout<<"VecY = "<<q.rotate( Vec3d(0.0,1.0,0.0) )<<std::endl;
    std::cout<<"VecZ = "<<q.rotate( Vec3d(0.0,0.0,1.0) )<<std::endl;


    // on veut maintenant retrouver l'Èquivalent de q_q0 dans le repËre global
    // c'est ‡ dire une rotation de Pi/2 autour de l'axe y
    Quat q_test = q * q0.inverse();

    std::cout<<"q_test = "<<q_test<<std::endl;

    std::cout<<"Alpha = "<<q_test.toEulerVector()<< " doit valoir une rotation de Pi/2 autour de l'axe y"<<std::endl;








}

template<>
PrecomputedConstraintCorrection<defaulttype::Vec3Types>::~PrecomputedConstraintCorrection()
{
    delete [] appCompliance;
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
    std::list<int> activeDof;
    for(unsigned int c1 = 0; c1 < numConstraints; c1++)
    {
        for(unsigned int i = 0; i < constraints[c1].size(); i++)
            activeDof.push_back(constraints[c1][i].index);
    }
    //unsigned int numNodes1 = activeDof.size();
    //std::cout<< "numNodes : avant = "<<numNodes1;
    activeDof.sort();
    activeDof.unique();
//	unsigned int numNodes = activeDof.size();
    //std::cout<< " apres = "<<numNodes<<std::endl;


    ////////////////////////////////////////////////////////////
    unsigned int offset, offset2;
    unsigned int i,j,ii,jj, curRowConst, curColConst, sizeCurColConst, sizeCurRowConst, it;
    Deriv Vbuf;
    int indexCurColConst, indexCurRowConst;
    it=0;

    //////////////////////////////////////////
    //std::vector<Deriv> sparseCompliance;
    std::list<int>::iterator IterateurListe;
    for(IterateurListe=activeDof.begin(); IterateurListe!=activeDof.end(); IterateurListe++)
    {
        int NodeIdx = (*IterateurListe);
        _indexNodeSparseCompliance[NodeIdx]=it;
        for(curColConst = 0; curColConst < numConstraints; curColConst++)
        {
            sizeCurColConst = constraints[curColConst].size();
            indexCurColConst = mstate->getConstraintId()[curColConst];

            const Const& c2 = constraints[curColConst];

            Vbuf.clear();
            for(j = 0; j < sizeCurColConst; j++)
            {
                const Deriv& n2 = c2[j].data;
                offset = dof_on_node*(NodeIdx * nbCols +  c2[j].index);

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
        sizeCurRowConst = constraints[curRowConst].size(); //number of nodes in constraint
        indexCurRowConst = mstate->getConstraintId()[curRowConst];//global index of constraint

        const Const& c1 = constraints[curRowConst];
        for(i = 0; i < sizeCurRowConst; i++)
        {

            const int NodeIdx  = c1[i].index;
            const Deriv& n1 = c1[i].data;

            unsigned int toto =(unsigned int) _indexNodeSparseCompliance[NodeIdx];

            for(curColConst = curRowConst; curColConst < numConstraints; curColConst++)
            {
                indexCurColConst = mstate->getConstraintId()[curColConst];
                double w = _sparseCompliance[toto + curColConst]*n1;
                //W[indexCurRowConst][indexCurColConst] += w;
                //std::cout << "W("<<indexCurRowConst<<","<<indexCurColConst<<") = "<<w<<std::endl;
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

//	std::cout<<"First list:"<<std::endl;
    for(unsigned int c1 = 0; c1 < numConstraints; c1++)
    {
        int indexC1 = mstate->getConstraintId()[c1];

        double fC1 = f->element(indexC1);
        //std::cout << "fC("<<indexC1<<")="<<fC1<<std::endl;

        if (fC1 != 0.0)
        {

            const int sizeC1 = constraints[c1].size();
            for(int i = 0; i < sizeC1; i++)
            {
                //on ne fait pas passer les forces du repere courant a celui initial ?
                // <-non, car elles ont deja ete tournees car on utilise une reference dans getCompliance !!!
                const Deriv& toto =  constraints[c1][i].data * fC1;
                force[constraints[c1][i].index] += toto;
                activeDof.push_back(constraints[c1][i].index);
            }
        }
    }

    activeDof.sort();
    activeDof.unique();

    //for (unsigned int i=0; i< force.size(); i++)
    //    std::cout << "f("<<i<<")="<<force[i]<<std::endl;

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
        //std::cout << "dx("<<i<<")="<<dx[i]<<std::endl;
        x[i] = x_free[i];
        v[i] = v_free[i];

        x[i] += dx[i];
        v[i] += dx[i]*(1/dt);
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
void PrecomputedConstraintCorrection<defaulttype::Vec1dTypes>::applyContactForce(double *f){


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

template<class DataTypes>
void PrecomputedConstraintCorrection<DataTypes>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->Inherit::parse(arg);
}

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec3fTypes>::rotateConstraints()
{
    VecConst& constraints = *mstate->getC();
    unsigned int numConstraints = constraints.size();

    simulation::tree::GNode *node = dynamic_cast<simulation::tree::GNode *>(getContext());

    sofa::component::forcefield::TetrahedronFEMForceField<defaulttype::Vec3fTypes>* forceField = NULL;
    if (node != NULL)
    {
//		core::componentmodel::behavior::BaseForceField* _forceField = node->forceField[1];
        forceField = node->get<component::forcefield::TetrahedronFEMForceField<defaulttype::Vec3fTypes> > ();
    }
    else
    {
        cout << "No rotation defined : only defined for TetrahedronFEMForceField !";
        return;
    }


    //std::cout << "start rotating normals " << g_timer_elapsed(timer, &micro) << std::endl;
//	int sizemax=0;
//	int index_const = -1;
    // on fait tourner les normales (en les ramenant dans le "pseudo" repere initial) //
    for(unsigned int curRowConst = 0; curRowConst < numConstraints; curRowConst++)
    {
        int sizeCurRowConst = constraints[curRowConst].size(); //number of nodes in constraint

        //Rmk : theres is one constraint for each contact direction, i.e. normal, tangent1, tangent2.
        for(int i = 0; i < sizeCurRowConst; i++)
        {
            const int localRowNodeIdx = constraints[curRowConst][i].index;
            Transformation Ri;
            forceField->getRotation(Ri, localRowNodeIdx);
            Ri.transpose();
            // on passe les normales du repere global au repere local
            const Deriv& n_i = Ri * constraints[curRowConst][i].data;
            constraints[curRowConst][i].data.x() =  n_i.x();
            constraints[curRowConst][i].data.y() =  n_i.y();
            constraints[curRowConst][i].data.z() =  n_i.z();
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
void PrecomputedConstraintCorrection<defaulttype::Vec3dTypes>::rotateConstraints()
{
    VecConst& constraints = *mstate->getC();
    unsigned int numConstraints = constraints.size();

    simulation::tree::GNode *node = dynamic_cast<simulation::tree::GNode *>(getContext());

    sofa::component::forcefield::TetrahedronFEMForceField<defaulttype::Vec3dTypes>* forceField = NULL;
    if (node != NULL)
    {
//		core::componentmodel::behavior::BaseForceField* _forceField = node->forceField[1];
        forceField = node->get<component::forcefield::TetrahedronFEMForceField<defaulttype::Vec3dTypes> > ();
    }
    else
    {
        cout << "No rotation defined : only defined for TetrahedronFEMForceField !";
        return;
    }


    //std::cout << "start rotating normals " << g_timer_elapsed(timer, &micro) << std::endl;
//	int sizemax=0;
//	int index_const = -1;
    // on fait tourner les normales (en les ramenant dans le "pseudo" repere initial) //
    for(unsigned int curRowConst = 0; curRowConst < numConstraints; curRowConst++)
    {
        int sizeCurRowConst = constraints[curRowConst].size(); //number of nodes in constraint

        //Rmk : theres is one constraint for each contact direction, i.e. normal, tangent1, tangent2.
        for(int i = 0; i < sizeCurRowConst; i++)
        {
            const int localRowNodeIdx = constraints[curRowConst][i].index;
            Transformation Ri;
            forceField->getRotation(Ri, localRowNodeIdx);
            Ri.transpose();
            // on passe les normales du repere global au repere local
            const Deriv& n_i = Ri * constraints[curRowConst][i].data;
            constraints[curRowConst][i].data.x() =  n_i.x();
            constraints[curRowConst][i].data.y() =  n_i.y();
            constraints[curRowConst][i].data.z() =  n_i.z();
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
void PrecomputedConstraintCorrection<defaulttype::Rigid3dTypes>::rotateConstraints()
{
    VecCoord& x = *mstate->getX();
    VecConst& constraints = *mstate->getC();
    VecCoord& x0 = *mstate->getX0();

    unsigned int numConstraints = constraints.size();
//	int sizemax=0;
//	int index_const = -1;
    // on fait tourner les normales (en les ramenant dans le "pseudo" repere initial) //
    for(unsigned int curRowConst = 0; curRowConst < numConstraints; curRowConst++)
    {
        int sizeCurRowConst = constraints[curRowConst].size(); //number of nodes in constraint

        for(int i = 0; i < sizeCurRowConst; i++)
        {
            const int localRowNodeIdx = constraints[curRowConst][i].index;
            Quat q;
            if (_restRotations)
                q = x[localRowNodeIdx].getOrientation() * x0[localRowNodeIdx].getOrientation().inverse();
            else
                q = x[localRowNodeIdx].getOrientation();


            Vec3d n_i = q.inverseRotate(constraints[curRowConst][i].data.getVCenter());
            Vec3d wn_i= q.inverseRotate(constraints[curRowConst][i].data.getVOrientation());

            // on passe les normales du repere global au repere local
            constraints[curRowConst][i].data.getVCenter() = n_i;
            constraints[curRowConst][i].data.getVOrientation() = wn_i;

        }
    }
}

template<>
void PrecomputedConstraintCorrection<defaulttype::Rigid3fTypes>::rotateConstraints()
{
    VecCoord& x = *mstate->getX();
    VecConst& constraints = *mstate->getC();
    VecCoord& x0 = *mstate->getX0();

    unsigned int numConstraints = constraints.size();
//	int sizemax=0;
//	int index_const = -1;
    // on fait tourner les normales (en les ramenant dans le "pseudo" repere initial) //
    for(unsigned int curRowConst = 0; curRowConst < numConstraints; curRowConst++)
    {
        int sizeCurRowConst = constraints[curRowConst].size(); //number of nodes in constraint

        for(int i = 0; i < sizeCurRowConst; i++)
        {
            const int localRowNodeIdx = constraints[curRowConst][i].index;
            Quat q;
            if (_restRotations)
                q = x[localRowNodeIdx].getOrientation() * x0[localRowNodeIdx].getOrientation().inverse();
            else
                q = x[localRowNodeIdx].getOrientation();


            Vec3d n_i = q.inverseRotate(constraints[curRowConst][i].data.getVCenter());
            Vec3d wn_i= q.inverseRotate(constraints[curRowConst][i].data.getVOrientation());

            // on passe les normales du repere global au repere local
            constraints[curRowConst][i].data.getVCenter() = n_i;
            constraints[curRowConst][i].data.getVOrientation() = wn_i;

        }
    }
}

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1dTypes>::rotateConstraints()
{
}
template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1fTypes>::rotateConstraints()
{
}


template<>
void PrecomputedConstraintCorrection<defaulttype::Vec3dTypes>::rotateResponse()
{
    simulation::tree::GNode *node = dynamic_cast<simulation::tree::GNode *>(getContext());

    sofa::component::forcefield::TetrahedronFEMForceField<defaulttype::Vec3dTypes>* forceField = NULL;
    if (node != NULL)
    {
//		core::componentmodel::behavior::BaseForceField* _forceField = node->forceField[1];
        forceField = node->get<component::forcefield::TetrahedronFEMForceField<defaulttype::Vec3dTypes> > ();
    }
    else
    {
        cout << "No rotation defined  !";
        return;
    }
    VecDeriv& dx = *mstate->getDx();
    for(unsigned int j = 0; j < dx.size(); j++)
    {
        Transformation Rj;
        forceField->getRotation(Rj, j);
        // on passe les deplacements du repere local au repere global
        const Deriv& toto = Rj * dx[j];
        dx[j] = toto;
    }
}
template<>
void PrecomputedConstraintCorrection<defaulttype::Vec3fTypes>::rotateResponse()
{
    simulation::tree::GNode *node = dynamic_cast<simulation::tree::GNode *>(getContext());

    sofa::component::forcefield::TetrahedronFEMForceField<defaulttype::Vec3fTypes>* forceField = NULL;
    if (node != NULL)
    {
//		core::componentmodel::behavior::BaseForceField* _forceField = node->forceField[1];
        forceField = node->get<component::forcefield::TetrahedronFEMForceField<defaulttype::Vec3fTypes> > ();
    }
    else
    {
        cout << "No rotation defined  !";
        return;
    }
    VecDeriv& dx = *mstate->getDx();
    for(unsigned int j = 0; j < dx.size(); j++)
    {
        Transformation Rj;
        forceField->getRotation(Rj, j);
        // on passe les deplacements du repere local au repere global
        const Deriv& toto = Rj * dx[j];
        dx[j] = toto;
    }
}



template<>
void PrecomputedConstraintCorrection<defaulttype::Rigid3dTypes>::rotateResponse()
{

    VecDeriv& dx = *mstate->getDx();
    VecCoord& x = *mstate->getX();
    VecCoord& x0 = *mstate->getX0();
    for(unsigned int j = 0; j < dx.size(); j++)
    {
        // on passe les deplacements du repere local (au repos) au repere global
        Deriv toto ;
        Quat q;
        if (_restRotations)
            q = x[j].getOrientation() * x0[j].getOrientation().inverse();
        else
            q = x[j].getOrientation();

        toto.getVCenter()		= q.rotate(dx[j].getVCenter());
        toto.getVOrientation()  = q.rotate(dx[j].getVOrientation());
        dx[j] = toto;
    }
}

template<>
void PrecomputedConstraintCorrection<defaulttype::Rigid3fTypes>::rotateResponse()
{

    VecDeriv& dx = *mstate->getDx();
    VecCoord& x = *mstate->getX();
    VecCoord& x0 = *mstate->getX0();
    for(unsigned int j = 0; j < dx.size(); j++)
    {
        // on passe les deplacements du repere local (au repos) au repere global
        Deriv toto ;
        Quat q;
        if (_restRotations)
            q = x[j].getOrientation() * x0[j].getOrientation().inverse();
        else
            q = x[j].getOrientation();

        toto.getVCenter()		= q.rotate(dx[j].getVCenter());
        toto.getVOrientation()  = q.rotate(dx[j].getVOrientation());
        dx[j] = toto;
    }
}

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1dTypes>::rotateResponse()
{
}
template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1fTypes>::rotateResponse()
{
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
