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
#include <sofa/component/mass/UniformMass.h>
#include <sofa/simulation/common/MechanicalVisitor.h>

//compliance computation include
#include <sofa/component/odesolver/CGImplicitSolver.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>

#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>


//#include <glib.h>
#include <sstream>
#include <list>

namespace sofa
{

namespace component
{

namespace constraint
{
#define	MAX_NUM_CONSTRAINT_PER_NODE 10000
#define EPS_UNITARY_FORCE 0.01

using namespace sofa::component::odesolver;
using namespace sofa::component::linearsolver;
using namespace sofa::simulation;

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
    appCompliance = new Real[nbRows * nbCols];


    double dt = this->getContext()->getDt();

    std::stringstream ss;

    ss << this->getContext()->getName() << ".comp";

    std::ifstream compFileIn(ss.str().c_str(), std::ifstream::binary);

    sout << "try to open : " << ss.str() << endl;

    if(compFileIn.good())
    {
        sout << "file open : " << ss.str() << " compliance being loaded" << endl;
        //complianceLoaded = true;
        compFileIn.read((char*)appCompliance, nbCols * nbRows*sizeof(double));
        compFileIn.close();
    }
    else
    {
        sout << "can not open : " << ss.str() << " compliance being built" << endl;

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

        CGImplicitSolver* odeSolver;
        EulerImplicitSolver* EulerSolver;
        CGLinearSolver<GraphScatteredMatrix,GraphScatteredVector>* linearSolver;

        this->getContext()->get(odeSolver);
        this->getContext()->get(EulerSolver);
        this->getContext()->get(linearSolver);

        if(odeSolver)
            sout << "use CGImplicitSolver " << sendl;
        else if(EulerSolver && linearSolver)
            sout << "use EulerImplicitSolver &  CGLinearSolver" << sendl;
        else
        {
            serr<<"PrecomputedContactCorrection must be associated with CGImplicitSolver or EulerImplicitSolver+CGLinearSolver for the precomputation\nNo Precomputation" << sendl;
            return;
        }




        ///////////////////////// CHANGE THE PARAMETERS OF THE SOLVER /////////////////////////////////
        double buf_tolerance, buf_threshold;
        int	   buf_maxIter;
        if(odeSolver)
        {
            buf_tolerance = (double) odeSolver->f_tolerance.getValue();
            buf_maxIter   = (int) odeSolver->f_maxIter.getValue();
            buf_threshold = (double) odeSolver->f_smallDenominatorThreshold.getValue();
            odeSolver->f_tolerance.setValue(1e-20);
            odeSolver->f_maxIter.setValue(500);
            odeSolver->f_smallDenominatorThreshold.setValue(1e-35);
        }
        else/* if(linearSolver) */
        {
            buf_tolerance = (double) linearSolver->f_tolerance.getValue();
            buf_maxIter   = (int) linearSolver->f_maxIter.getValue();
            buf_threshold = (double) linearSolver->f_smallDenominatorThreshold.getValue();
            linearSolver->f_tolerance.setValue(1e-20);
            linearSolver->f_maxIter.setValue(500);
            linearSolver->f_smallDenominatorThreshold.setValue(1e-35);
        }
        ///////////////////////////////////////////////////////////////////////////////////////////////

        VecDeriv& velocity = *mstate->getV();
        VecCoord& pos=*mstate->getX();
        VecCoord& pos0=*mstate->getX0();



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
                }

                //serr<<"solve reussi"<<sendl;

                velocity = *mstate->getV();
                //serr<<"getV : "<<velocity<<sendl;
                for (unsigned int v=0; v<nbNodes; v++)
                {

                    for (unsigned int j=0; j<dof_on_node; j++)
                    {
                        appCompliance[(v*dof_on_node+j)*nbCols + (f*dof_on_node+i) ] = velocity[v][j] / unitary_force[i];
                    }
                }
                //serr<<"put in appComp"<<sendl;
            }
            unitary_force.clear();
            force[f] = unitary_force;
        }
        ///////////////////////// RESET PARAMETERS AT THEIR PREVIOUS VALUE /////////////////////////////////
        // gravity is reset at its previous value
        this->getContext()->setGravityInWorld(gravity);
        if(odeSolver)
        {
            odeSolver->f_tolerance.setValue(buf_tolerance);
            odeSolver->f_maxIter.setValue(buf_maxIter);
            odeSolver->f_smallDenominatorThreshold.setValue(buf_threshold);
        }
        else/* if(linearSolver) */
        {
            linearSolver->f_tolerance.setValue(buf_tolerance);
            linearSolver->f_maxIter.setValue(buf_maxIter);
            linearSolver->f_smallDenominatorThreshold.setValue(buf_threshold);
        }
        ///////////////////////////////////////////////////////////////////////////////////////////////
        std::ofstream compFileOut(ss.str().c_str(), std::fstream::out | std::fstream::binary);
        compFileOut.write((char*)appCompliance, nbCols * nbRows*sizeof(double));
        compFileOut.close();
    }

    // Optimisation for the computation of W
    _indexNodeSparseCompliance = new int[v0.size()];
    _sparseCompliance.resize(v0.size()*MAX_NUM_CONSTRAINT_PER_NODE);


    ////  debug print 100 first row and column of the matrix
    sout << "Matrix compliance" ;

    for (unsigned int i=0; i<10 && i<nbCols; i++)
    {
        sout << sendl;
        for (unsigned int j=0; j<10 && j<nbCols; j++)
        {
            sout <<" \t "<< appCompliance[j*nbCols + i];
        }
    }

    sout << sendl;
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
        ConstraintIterator itConstraint;
        for (itConstraint=constraints[c1].getData().begin(); itConstraint!=constraints[c1].getData().end(); itConstraint++)
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
            indexCurColConst = mstate->getConstraintId()[curColConst];

            Vbuf.clear();
            ConstraintIterator itConstraint;
            for (itConstraint=constraints[curColConst].getData().begin(); itConstraint!=constraints[curColConst].getData().end(); itConstraint++)
            {
                unsigned int dof = itConstraint->first;
                Deriv n = itConstraint->second;
                const Deriv& n2 = n;
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

        ConstraintIterator itConstraint;
        for (itConstraint=constraints[curRowConst].getData().begin(); itConstraint!=constraints[curRowConst].getData().end(); itConstraint++)
        {
            unsigned int dof = itConstraint->first;
            Deriv n = itConstraint->second;

            const int NodeIdx  = dof;
            const Deriv& n1 = n;

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

            ConstraintIterator itConstraint;
            for (itConstraint=constraints[c1].getData().begin(); itConstraint!=constraints[c1].getData().end(); itConstraint++)
            {
                unsigned int dof = itConstraint->first;
                Deriv n = itConstraint->second;
                //on ne fait pas passer les forces du repere courant a celui initial ?
                // <-non, car elles ont deja ete tournees car on utilise une reference dans getCompliance !!!
                const Deriv& temp =  n * fC1;
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

#ifndef SOFA_FLOAT

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec3dTypes>::rotateConstraints()
{
    VecConst& constraints = *mstate->getC();
    unsigned int numConstraints = constraints.size();

    simulation::Node *node = dynamic_cast<simulation::Node *>(getContext());

    sofa::component::forcefield::TetrahedronFEMForceField<defaulttype::Vec3dTypes>* forceField = NULL;
    if (node != NULL)
    {
        //		core::componentmodel::behavior::BaseForceField* _forceField = node->forceField[1];
        forceField = node->get<component::forcefield::TetrahedronFEMForceField<defaulttype::Vec3dTypes> > ();
    }
    else
    {
        sout << "No rotation defined : only defined for TetrahedronFEMForceField !";
        return;
    }


    //sout << "start rotating normals " << g_timer_elapsed(timer, &micro) << sendl;
    //	int sizemax=0;
    //	int index_const = -1;
    // on fait tourner les normales (en les ramenant dans le "pseudo" repere initial) //
    for(unsigned int curRowConst = 0; curRowConst < numConstraints; curRowConst++)
    {
        ConstraintIterator itConstraint;
        for (itConstraint=constraints[curRowConst].getData().begin(); itConstraint!=constraints[curRowConst].getData().end(); itConstraint++)
        {
            unsigned int dof = itConstraint->first;
            Deriv n = itConstraint->second;
            const int localRowNodeIdx = dof;
            Transformation Ri;
            forceField->getRotation(Ri, localRowNodeIdx);
            Ri.transpose();
            // on passe les normales du repere global au repere local
            const Deriv& n_i = Ri * n;
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
        ConstraintIterator itConstraint;
        for (itConstraint=constraints[curRowConst].getData().begin(); itConstraint!=constraints[curRowConst].getData().end(); itConstraint++)
        {
            unsigned int dof = itConstraint->first;
            Deriv n = itConstraint->second;
            const int localRowNodeIdx = dof;
            Quat q;
            if (_restRotations)
                q = x[localRowNodeIdx].getOrientation() * x0[localRowNodeIdx].getOrientation().inverse();
            else
                q = x[localRowNodeIdx].getOrientation();


            Vec3d n_i = q.inverseRotate(n.getVCenter());
            Vec3d wn_i= q.inverseRotate(n.getVOrientation());

            // on passe les normales du repere global au repere local
            n.getVCenter() = n_i;
            n.getVOrientation() = wn_i;

        }
    }
}



template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1dTypes>::rotateConstraints()
{
}


template<>
void PrecomputedConstraintCorrection<defaulttype::Vec3dTypes>::rotateResponse()
{
    simulation::Node *node = dynamic_cast<simulation::Node *>(getContext());

    sofa::component::forcefield::TetrahedronFEMForceField<defaulttype::Vec3dTypes>* forceField = NULL;
    if (node != NULL)
    {
        //		core::componentmodel::behavior::BaseForceField* _forceField = node->forceField[1];
        forceField = node->get<component::forcefield::TetrahedronFEMForceField<defaulttype::Vec3dTypes> > ();
    }
    else
    {
        sout << "No rotation defined  !";
        return;
    }
    VecDeriv& dx = *mstate->getDx();
    for(unsigned int j = 0; j < dx.size(); j++)
    {
        Transformation Rj;
        forceField->getRotation(Rj, j);
        // on passe les deplacements du repere local au repere global
        const Deriv& temp = Rj * dx[j];
        dx[j] = temp;
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
        Deriv temp ;
        Quat q;
        if (_restRotations)
            q = x[j].getOrientation() * x0[j].getOrientation().inverse();
        else
            q = x[j].getOrientation();

        temp.getVCenter()		= q.rotate(dx[j].getVCenter());
        temp.getVOrientation()  = q.rotate(dx[j].getVOrientation());
        dx[j] = temp;
    }
}

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1dTypes>::rotateResponse()
{
}
#endif
#ifndef SOFA_DOUBLE
template<>
void PrecomputedConstraintCorrection<defaulttype::Vec3fTypes>::rotateConstraints()
{
    VecConst& constraints = *mstate->getC();
    unsigned int numConstraints = constraints.size();

    simulation::Node *node = dynamic_cast<simulation::Node *>(getContext());

    sofa::component::forcefield::TetrahedronFEMForceField<defaulttype::Vec3fTypes>* forceField = NULL;
    if (node != NULL)
    {
        //		core::componentmodel::behavior::BaseForceField* _forceField = node->forceField[1];
        forceField = node->get<component::forcefield::TetrahedronFEMForceField<defaulttype::Vec3fTypes> > ();
    }
    else
    {
        sout << "No rotation defined : only defined for TetrahedronFEMForceField !";
        return;
    }


    //sout << "start rotating normals " << g_timer_elapsed(timer, &micro) << sendl;
    //	int sizemax=0;
    //	int index_const = -1;
    // on fait tourner les normales (en les ramenant dans le "pseudo" repere initial) //
    for(unsigned int curRowConst = 0; curRowConst < numConstraints; curRowConst++)
    {
        ConstraintIterator itConstraint;
        for (itConstraint=constraints[curRowConst].getData().begin(); itConstraint!=constraints[curRowConst].getData().end(); itConstraint++)
        {
            unsigned int dof = itConstraint->first;
            Deriv n = itConstraint->second;
            const int localRowNodeIdx = dof;
            Transformation Ri;
            forceField->getRotation(Ri, localRowNodeIdx);
            Ri.transpose();
            // on passe les normales du repere global au repere local
            const Deriv& n_i = Ri * n;
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
        ConstraintIterator itConstraint;
        for (itConstraint=constraints[curRowConst].getData().begin(); itConstraint!=constraints[curRowConst].getData().end(); itConstraint++)
        {
            unsigned int dof = itConstraint->first;
            Deriv n = itConstraint->second;
            const int localRowNodeIdx = dof;
            Quat q;
            if (_restRotations)
                q = x[localRowNodeIdx].getOrientation() * x0[localRowNodeIdx].getOrientation().inverse();
            else
                q = x[localRowNodeIdx].getOrientation();


            Vec3d n_i = n.getVCenter();
            Vec3d wn_i= n.getVOrientation();

            // on passe les normales du repere global au repere local
            n.getVCenter() = n_i;
            n.getVOrientation() = wn_i;

        }
    }
}

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1fTypes>::rotateConstraints()
{
}


template<>
void PrecomputedConstraintCorrection<defaulttype::Vec3fTypes>::rotateResponse()
{
    simulation::Node *node = dynamic_cast<simulation::Node *>(getContext());

    sofa::component::forcefield::TetrahedronFEMForceField<defaulttype::Vec3fTypes>* forceField = NULL;
    if (node != NULL)
    {
        //		core::componentmodel::behavior::BaseForceField* _forceField = node->forceField[1];
        forceField = node->get<component::forcefield::TetrahedronFEMForceField<defaulttype::Vec3fTypes> > ();
    }
    else
    {
        sout << "No rotation defined  !";
        return;
    }
    VecDeriv& dx = *mstate->getDx();
    for(unsigned int j = 0; j < dx.size(); j++)
    {
        Transformation Rj;
        forceField->getRotation(Rj, j);
        // on passe les deplacements du repere local au repere global
        const Deriv& temp = Rj * dx[j];
        dx[j] = temp;
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
        Deriv temp ;
        Quat q;
        if (_restRotations)
            q = x[j].getOrientation() * x0[j].getOrientation().inverse();
        else
            q = x[j].getOrientation();

        temp.getVCenter()		= q.rotate(dx[j].getVCenter());
        temp.getVOrientation()  = q.rotate(dx[j].getVOrientation());
        dx[j] = temp;
    }
}

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1fTypes>::rotateResponse()
{
}

#endif



} // namespace collision

} // namespace component

} // namespace sofa

#endif
