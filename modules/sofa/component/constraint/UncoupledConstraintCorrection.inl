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
#ifndef SOFA_COMPONENT_CONSTRAINT_UNCOUPLEDCONSTRAINTCORRECTION_INL
#define SOFA_COMPONENT_CONSTRAINT_UNCOUPLEDCONSTRAINTCORRECTION_INL

#include "UncoupledConstraintCorrection.h"
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/component/mass/UniformMass.h>
#include <sofa/simulation/common/MechanicalVisitor.h>

namespace sofa
{

namespace component
{

namespace constraint
{

template<class DataTypes>
UncoupledConstraintCorrection<DataTypes>::UncoupledConstraintCorrection(behavior::MechanicalState<DataTypes> *mm)
    : mstate(mm)
    ,compliance(initData(&compliance, "compliance", "compliance value on each dof"))
{
}

template<class DataTypes>
UncoupledConstraintCorrection<DataTypes>::~UncoupledConstraintCorrection()
{
}
/*
template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::init()
{
    mstate = dynamic_cast< behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
	const VecCoord& x = *mstate->getX();

	if (x.size() != compliance.getValue().size())
	{
		sout<<"Warning compliance size is not the size of the mstate"<<sendl;
		VecReal UsedComp;
		if (compliance.getValue().size()>0)
		{
			for (unsigned int i=0; i<x.size(); i++)
			{
				UsedComp.push_back(compliance.getValue()[0]);
			}
		}
		else
		{
			for (unsigned int i=0; i<x.size(); i++)
			{
				Real random_value = (Real)0.00001;
				UsedComp.push_back(random_value);
			}
		}
		compliance.setValue(UsedComp);
	}
}
*/


template<>
void UncoupledConstraintCorrection<defaulttype::Vec1Types>::init()
{
    mstate = dynamic_cast< behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    const VecCoord& x = *mstate->getX();

    if (x.size() != compliance.getValue().size())
    {
        serr<<"Warning compliance size is not the size of the mstate"<<sendl;
        VecReal UsedComp;
        if (compliance.getValue().size()>0)
        {
            for (unsigned int i=0; i<x.size(); i++)
            {
                UsedComp.push_back(compliance.getValue()[0]);
            }
        }
        else
        {
            for (unsigned int i=0; i<x.size(); i++)
            {
                Real random_value = (Real)0.00001;
                UsedComp.push_back(random_value);
            }
        }
        compliance.setValue(UsedComp);
    }
}


template<>
void UncoupledConstraintCorrection<defaulttype::Rigid3Types>::init()
{
    mstate = dynamic_cast< behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    const sofa::defaulttype::Rigid3Mass* massValue;

    simulation::Node *node = dynamic_cast<simulation::Node *>(getContext());

    if (node != NULL)
    {
        core::componentmodel::behavior::BaseMass*_m = node->mass;
        component::mass::UniformMass<defaulttype::Rigid3Types, defaulttype::Rigid3Mass> *m = dynamic_cast<component::mass::UniformMass<defaulttype::Rigid3Types, defaulttype::Rigid3Mass>*> (_m);
        massValue = &( m->getMass());
    }
    else
    {
        massValue = new sofa::defaulttype::Rigid3Mass();
        serr<<"\n WARNING : node is not found => massValue could be false in getCompliance function"<<sendl;
    }


    double dt = this->getContext()->getDt();



    VecReal UsedComp;

    UsedComp.push_back(dt*dt/massValue->mass);
    UsedComp.push_back(dt*dt*massValue->invInertiaMassMatrix[0][0]);
    UsedComp.push_back(dt*dt*massValue->invInertiaMassMatrix[0][1]);
    UsedComp.push_back(dt*dt*massValue->invInertiaMassMatrix[0][2]);
    UsedComp.push_back(dt*dt*massValue->invInertiaMassMatrix[1][1]);
    UsedComp.push_back(dt*dt*massValue->invInertiaMassMatrix[1][2]);
    UsedComp.push_back(dt*dt*massValue->invInertiaMassMatrix[2][2]);
    compliance.setValue(UsedComp);

}

template<>
void UncoupledConstraintCorrection<defaulttype::Rigid3Types>::getCompliance(defaulttype::BaseMatrix*W)
{
    const VecConst& constraints = *mstate->getC();
    Deriv weighedNormal;
    Deriv InvM_wN;

    const sofa::defaulttype::Rigid3Mass* massValue;

    simulation::Node *node = dynamic_cast<simulation::Node *>(getContext());

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


    //sout<<"Mass Value  = "<< massValue[1] <<sendl;
    unsigned int numConstraints = constraints.size();
    double dt = this->getContext()->getDt();

    for(unsigned int curRowConst = 0; curRowConst < numConstraints; curRowConst++)
    {
        int indexCurRowConst = mstate->getConstraintId()[curRowConst];

        //sout<<"constraint["<<curRowConst<<"] : ";



        ConstraintIterator itConstraint;
        std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[curRowConst].data();

        for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
        {
            unsigned int dof = itConstraint->first;
            Deriv n = itConstraint->second;

            weighedNormal.getVCenter() = n.getVCenter(); // weighed normal
            weighedNormal.getVOrientation() = n.getVOrientation();

            //sout<<" - "<<weighedNormal;

            InvM_wN = weighedNormal / (*massValue);
            InvM_wN *= dt*dt ;

            int indexCurColConst;

            for(unsigned int curColConst = curRowConst; curColConst < numConstraints; curColConst++)
            {
                indexCurColConst = mstate->getConstraintId()[curColConst];

                ConstraintIterator itConstraint2;
                std::pair< ConstraintIterator, ConstraintIterator > iter2=constraints[curColConst].data();

                for (itConstraint2=iter2.first; itConstraint2!=iter2.second; itConstraint2++)
                {
                    unsigned int dof2 = itConstraint2->first;
                    Deriv n2 = itConstraint2->second;
                    //W[indexCurRowConst][indexCurColConst] +=  constraints[curColConst][j].data * InvM_wN;

                    if (dof == dof2)
                    {
                        double w =  n2 * InvM_wN;
                        W->add(indexCurRowConst, indexCurColConst, w);
                        if (indexCurRowConst != indexCurColConst)
                            W->add(indexCurColConst, indexCurRowConst, w);
                    }
                }
            }

            /*
                  for(unsigned int curColConst = curRowConst+1; curColConst < numConstraints; curColConst++)
                  {
                  indexCurColConst = mstate->getConstraintId()[curColConst];
                  W[indexCurColConst][indexCurRowConst] = W[indexCurRowConst][indexCurColConst];
                }
            */
        }
        //sout<<" end"<<sendl;
    }
}

template<>
void UncoupledConstraintCorrection<defaulttype::Rigid3Types>::applyContactForce(const defaulttype::BaseVector *f)
{
    VecDeriv& force = *mstate->getExternalForces();
    const VecConst& constraints = *mstate->getC();
    Deriv weighedNormal;

    const sofa::defaulttype::Rigid3Mass* massValue;

    simulation::Node *node = dynamic_cast<simulation::Node *>(getContext());

    if (node != NULL)
    {
        core::componentmodel::behavior::BaseMass*_m = node->mass;
        component::mass::UniformMass<defaulttype::Rigid3Types, defaulttype::Rigid3Mass> *m = dynamic_cast<component::mass::UniformMass<defaulttype::Rigid3Types, defaulttype::Rigid3Mass>*> (_m);
        massValue = &( m->getMass());
    }
    else
    {
        massValue = new sofa::defaulttype::Rigid3Mass();
        printf("\n WARNING : node is not found => massValue could be false in applyContactForce function");
    }


    double dt = this->getContext()->getDt();

    //force.resize(0);
    //force.resize(1);
    //force[0] = Deriv();
    force.resize((*mstate->getX()).size());

    int numConstraints = constraints.size();

    for(int c1 = 0; c1 < numConstraints; c1++)
    {
        int indexC1 = mstate->getConstraintId()[c1];
        double fC1 = f->element(indexC1);

        if (fC1 != 0.0)
        {
            ConstraintIterator itConstraint;
            std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[c1].data();

            for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
            {
                unsigned int dof = itConstraint->first;
                weighedNormal = itConstraint->second; // weighted normal
                force[dof].getVCenter() += weighedNormal.getVCenter() * fC1;
                force[dof].getVOrientation() += weighedNormal.getVOrientation() * fC1;
            }
        }
    }


    VecDeriv& dx = *mstate->getDx();
    VecCoord& x = *mstate->getX();
    VecDeriv& v = *mstate->getV();
    VecDeriv& v_free = *mstate->getVfree();
    VecCoord& x_free = *mstate->getXfree();

// Euler integration... will be done in the "integrator" as soon as it exists !
    dx.resize(v.size());

    for (unsigned int i=0; i<dx.size(); i++)
    {
        x[i] = x_free[i];
        v[i] = v_free[i];
        dx[i] = force[i] / (*massValue);
        dx[i] *= dt;
        v[i] += dx[i];
        dx[i] *= dt;
        x[i] += dx[i];
    }

//	simulation::tree::MechanicalPropagateAndAddDxVisitor(dx).execute(this->getContext());


////////////////////////////////////////////////////////////////////










}

template<>
void UncoupledConstraintCorrection<defaulttype::Vec1Types>::getCompliance(defaulttype::BaseMatrix *W)
{
    const VecConst& constraints = *mstate->getC();
    unsigned int numConstraints = constraints.size();

    for(unsigned int curRowConst = 0; curRowConst < numConstraints; curRowConst++)
    {

        int indexCurRowConst = mstate->getConstraintId()[curRowConst];
        //sout<<"constraint["<<curRowConst<<"] : ";
        ConstraintIterator itConstraint;
        std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[curRowConst].data();

        for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
        {
            unsigned int dof = itConstraint->first;
            Deriv n = itConstraint->second;

            int indexCurColConst;
            //  sout<<" : "<<constraints[curRowConst][i].data.x();

            for(unsigned int curColConst = curRowConst; curColConst < numConstraints; curColConst++)
            {

                indexCurColConst = mstate->getConstraintId()[curColConst];

                ConstraintIterator itConstraint2;
                std::pair< ConstraintIterator, ConstraintIterator > iter2=constraints[curColConst].data();

                for (itConstraint2=iter2.first; itConstraint2!=iter2.second; itConstraint2++)
                {
                    unsigned int dof2 = itConstraint->first;
                    Deriv n2 = itConstraint2->second;
                    if (dof == dof2)
                    {
                        //W[indexCurRowConst][indexCurColConst] += (1.0/10000.0) * constraints[curRowConst][i].data.x() * constraints[curColConst][j].data.x();
                        double w = compliance.getValue()[dof] * n.x() * n2.x();
                        W->add(indexCurRowConst, indexCurColConst, w);
                        if (indexCurRowConst != indexCurColConst)
                            W->add(indexCurColConst, indexCurRowConst, w);
                    }
                }
            }
            /*
                  for(unsigned int curColConst = curRowConst+1; curColConst < numConstraints; curColConst++)
                  {
                  indexCurColConst = mstate->getConstraintId()[curColConst];
                  W[indexCurColConst][indexCurRowConst] = W[indexCurRowConst][indexCurColConst];
                }
            */
        }
        //sout<<" : "<<sendl;
    }

// debug : verifie qu'il n'y a pas de 0 sur la diagonale de W
    //printf("\n index : ");
    //for(unsigned int curRowConst = 0; curRowConst < numConstraints; curRowConst++)
    //{
    //	int indexCurRowConst = mstate->getConstraintId()[curRowConst];
    //	printf(" %d ",indexCurRowConst);
    //	if(abs(W[indexCurRowConst][indexCurRowConst]) < 0.000000001)
    //		printf("\n WARNING : there is a 0 on the diagonal of matrix W");

    //	if(abs(W[curRowConst][curRowConst]) <0.000000001)
    //		printf("\n stop");
    //}


}

template<>
void UncoupledConstraintCorrection<defaulttype::Vec1Types>::applyContactForce(const defaulttype::BaseVector *f)
{

    VecDeriv& force = *mstate->getExternalForces();
    const VecConst& constraints = *mstate->getC();
    unsigned int numConstraints = constraints.size();

    force.resize((*mstate->getX()).size());

    for(unsigned int c1 = 0; c1 < numConstraints; c1++)
    {
        int indexC1 = mstate->getConstraintId()[c1];
        double fC1 = f->element(indexC1);
        if (fC1 != 0.0)
        {
            ConstraintIterator itConstraint;
            std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[c1].data();

            for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
            {
                unsigned int dof = itConstraint->first;
                Deriv n = itConstraint->second;
                force[dof] += n * fC1;
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
        dx[i] = force[i] *compliance.getValue()[i];
        x[i] += dx[i];
        v[i] += dx[i]/dt;
    }
    //sout<<" dx on articulations"<<dx<<sendl;
}


template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::resetContactForce()
{
    VecDeriv& force = *mstate->getExternalForces();
    for( unsigned i=0; i<force.size(); ++i )
        force[i] = Deriv();
}


///////////////////////  new API for non building the constraint system during solving process //
template<class DataTypes>
bool UncoupledConstraintCorrection<DataTypes>::hasConstraintNumber(int index)
{
    const VecConst& constraints = *mstate->getC();
    unsigned int numConstraints = constraints.size();

    for(unsigned int c = 0; c < numConstraints; c++)
    {
        int indexC = mstate->getConstraintId()[c];
        if (indexC == index)
            return true;
    }
    return false;
}


template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::resetForUnbuiltResolution(double * f, std::list<int>& /*renumbering*/)
{



    const VecConst& constraints = *mstate->getC();
    unsigned int numConstraints = constraints.size();

    constraint_disp.clear();
    constraint_disp.resize(mstate->getSize());

    constraint_force.clear();
    constraint_force.resize(mstate->getSize());

    constraint_dofs.clear();
    id_to_localIndex.clear();


    int maxIndex = -1;
    for(unsigned int c = 0; c < numConstraints; c++)
    {
        int indexC = mstate->getConstraintId()[c];

        // resize table if necessary
        if (indexC > maxIndex)
        {
            id_to_localIndex.resize(indexC+1, -1);   // debug : -1 value allows to know if the table is badly filled
            maxIndex = indexC;
        }
        // buf the table of local indices
        id_to_localIndex[indexC] = c;


        // buf the value of force applied on concerned dof : constraint_force
        // buf a table of indice of involved dof : constraint_dofs
        double fC = f[indexC];
        // debug
        //std::cout<<"f["<<indexC<<"] = "<<fC<<std::endl;

        if (fC != 0.0)
        {
            ConstraintIterator itConstraint;
            std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[c].data();

            for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
            {

                unsigned int dof = itConstraint->first;
                Deriv n = itConstraint->second;
                constraint_force[dof] +=n * fC;
                constraint_dofs.push_back(dof);
            }
        }
    }

    // debug
    //std::cout<<"in resetConstraintForce : constraint_force ="<<constraint_force<<std::endl;

    // constraint_dofs buff the DOF that are involved with the constraints
    constraint_dofs.unique();
}

template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::addConstraintDisplacement(double * d, int begin, int end)
{

/// in the Vec1Types case, compliance is a vector of size mstate->getSize()
/// constraint_force contains the force applied on dof involved with the contact
/// TODO : compute a constraint_disp that is updated each time a new force is provided !



    const VecConst& constraints = *mstate->getC();

    for (int id_=begin; id_<=end; id_++)
    {
        int c = id_to_localIndex[id_];

        ConstraintIterator itConstraint;
        std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[c].data();

        for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
        {
            //Deriv DX =  constraint_force[constraints[c][i].index] * compliance.getValue()[constraints[c][i].index];
            Deriv n = itConstraint->second;
            d[id_] += n * constraint_disp[itConstraint->first];
        }

    }

}



template<>
void UncoupledConstraintCorrection<defaulttype::Vec1Types>::setConstraintDForce(double * df, int begin, int end, bool update)
{
    const VecConst& constraints = *mstate->getC();

    if (!update)
        return;
    // debug
    //if (end<6)
    //	std::cout<<"addDf - df["<<begin<<" to "<<end<<"] ="<< df[begin] << " " << df[begin+1] << " "<< df[begin+2] << std::endl;

    for ( int id_=begin; id_<=end; id_++)
    {
        int c = id_to_localIndex[id_];

        ConstraintIterator itConstraint;
        std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[c].data();
        for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
        {
            Deriv n = itConstraint->second;
            unsigned int dof = itConstraint->first;

            constraint_force[dof] += n * df[id_];

            Deriv DX =  constraint_force[dof] * compliance.getValue()[dof];

            constraint_disp[dof] = DX;
        }

    }

}

template<>
void UncoupledConstraintCorrection<defaulttype::Rigid3Types>::setConstraintDForce(double * df, int begin, int end, bool update)
{
    const VecConst& constraints = *mstate->getC();
    if (!update)
        return;
    // debug
    //if (end<6)
    //	std::cout<<"addDf - df["<<begin<<" to "<<end<<"] ="<< df[begin] << " " << df[begin+1] << " "<< df[begin+2] << std::endl;

    for ( int id_=begin; id_<=end; id_++)
    {
        int c = id_to_localIndex[id_];
        ConstraintIterator itConstraint;
        std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[c].data();
        for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
        {

            Deriv n = itConstraint->second;
            unsigned int dof = itConstraint->first;

            constraint_force[dof] += n * df[id_];

            Deriv DX;
            DX.getVCenter() = constraint_force[dof].getVCenter() * compliance.getValue()[0];
            defaulttype::Vec3d wrench = constraint_force[dof].getVOrientation();
            DX.getVOrientation()[0] = compliance.getValue()[1]*wrench[0] +  compliance.getValue()[2]*wrench[1] + compliance.getValue()[3]*wrench[2] ;
            DX.getVOrientation()[1] = compliance.getValue()[2]*wrench[0] +  compliance.getValue()[4]*wrench[1] + compliance.getValue()[5]*wrench[2] ;
            DX.getVOrientation()[2] = compliance.getValue()[3]*wrench[0] +  compliance.getValue()[5]*wrench[1] + compliance.getValue()[6]*wrench[2] ;


            constraint_disp[dof] = DX;
        }

    }
}


template<>
void UncoupledConstraintCorrection<defaulttype::Vec1Types>::getBlockDiagonalCompliance(defaulttype::BaseMatrix* W, int begin, int end)
{

    const VecConst& constraints = *mstate->getC();


    for (int id1=begin; id1<=end; id1++)
    {
        int c1 = id_to_localIndex[id1];
        ConstraintIterator itConstraint1;
        std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[c1].data();
        for (itConstraint1=iter.first; itConstraint1!=iter.second; itConstraint1++)
        {
            Deriv n1 = itConstraint1->second;
            unsigned int dof1 = itConstraint1->first;

            for (int id2= id1; id2<=end; id2++)
            {
                int c2 = id_to_localIndex[id2];

                ConstraintIterator itConstraint2;
                std::pair< ConstraintIterator, ConstraintIterator > iter2=constraints[c2].data();
                for (itConstraint2=iter2.first; itConstraint2!=iter2.second; itConstraint2++)
                {

                    unsigned int dof2 = itConstraint2->first;

                    if (dof1 == dof2)
                    {
                        Deriv n2 = itConstraint2->second;
                        double w = compliance.getValue()[dof1] * n1.x() * n2.x();
                        W->add(id1, id2, w);
                        if (id1 != id2)
                            W->add(id2, id1, w);

                    }
                }
            }
        }
    }
}


///////////////////// ATTENTION : passer un indice début - fin (comme pour force et déplacement) pour calculer le block complet
///////////////////// et pas uniquement la diagonale.
template<>
void UncoupledConstraintCorrection<defaulttype::Rigid3Types>::getBlockDiagonalCompliance(defaulttype::BaseMatrix* W, int begin, int end)
{

    //std::cout<<"UncoupledConstraintCorrection<defaulttype::Rigid3Types>::getBlockDiagonalCompliance"<<std::endl;

    const VecConst& constraints = *mstate->getC();

    //std::cout<<" begin = "<<begin<<"  - end = "<<std::endl;

    Deriv weighedNormal, C_n;

    //std::cerr<<" weighedNormal, C_n "<<std::endl;

    for (int id1=begin; id1<=end; id1++)
    {
        //std::cerr<<"constraint : "<<id1;
        int c1 = id_to_localIndex[id1];
        //std::cerr<<" local index : "<<c1<<std::endl;

        ConstraintIterator itConstraint1;
        std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[c1].data();

        for (itConstraint1=iter.first; itConstraint1!=iter.second; itConstraint1++)
        {
            weighedNormal = itConstraint1->second;
            unsigned int dof1 = itConstraint1->first;


            C_n.getVCenter() = weighedNormal.getVCenter() * compliance.getValue()[0];
            defaulttype::Vec3d wrench = weighedNormal.getVOrientation() ;
            C_n.getVOrientation()[0] = compliance.getValue()[1]*wrench[0] +  compliance.getValue()[2]*wrench[1] + compliance.getValue()[3]*wrench[2] ;
            C_n.getVOrientation()[1] = compliance.getValue()[2]*wrench[0] +  compliance.getValue()[4]*wrench[1] + compliance.getValue()[5]*wrench[2] ;
            C_n.getVOrientation()[2] = compliance.getValue()[3]*wrench[0] +  compliance.getValue()[5]*wrench[1] + compliance.getValue()[6]*wrench[2] ;

            //std::cout<<"C_n : "<<C_n<<std::endl;

            for (int id2= id1; id2<=end; id2++)
            {
                int c2 = id_to_localIndex[id2];
                ConstraintIterator itConstraint2;
                std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[c2].data();
                for (itConstraint2=iter.first; itConstraint2!=iter.second; itConstraint2++)
                {

                    unsigned int dof2 = itConstraint2->first;

                    if (dof1 == dof2)
                    {
                        Deriv n2 = itConstraint2->second;
                        double w = n2 * C_n;
                        // debug
                        //std::cout<<"W("<<id1<<","<<id2<<") += "<< w ;
                        W->add(id1, id2, w);
                        if (id1 != id2)
                            W->add(id2, id1, w);
                    }
                }
            }
        }
    }

}




} // namespace constraint

} // namespace component

} // namespace sofa

#endif
