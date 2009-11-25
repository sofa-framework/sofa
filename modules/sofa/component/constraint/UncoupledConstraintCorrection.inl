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


template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::init()
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


template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::getCompliance(defaulttype::BaseMatrix *W)
{
    const VecConst& constraints = *mstate->getC();
    unsigned int numConstraints = constraints.size();
    //std::cout<<"UncoupledConstraintCorrection ("<<this->getName()<<")::getCompliance is called on "<< mstate->getName()<<std::endl;
    //std::cout<<"numConstraints ="<<numConstraints<<std::endl;

    for(unsigned int curRowConst = 0; curRowConst < numConstraints; curRowConst++)
    {

        int indexCurRowConst = mstate->getConstraintId()[curRowConst];
        //std::cout<<"constraint["<<curRowConst<<"] : ";
        ConstraintIterator itConstraint;
        std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[curRowConst].data();

        for (itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
        {
            unsigned int dof = itConstraint->first;
            Deriv n = itConstraint->second;

            int indexCurColConst;
            //std::cout<<" [ "<<dof<<"]="<<n<<std::endl;

            for(unsigned int curColConst = curRowConst; curColConst < numConstraints; curColConst++)
            {

                indexCurColConst = mstate->getConstraintId()[curColConst];

                ConstraintIterator itConstraint2;
                std::pair< ConstraintIterator, ConstraintIterator > iter2=constraints[curColConst].data();

                for (itConstraint2=iter2.first; itConstraint2!=iter2.second; itConstraint2++)
                {
                    unsigned int dof2 = itConstraint2->first;
                    Deriv n2 = itConstraint2->second;
                    if (dof == dof2)
                    {
                        //W[indexCurRowConst][indexCurColConst] += (1.0/10000.0) * constraints[curRowConst][i].data.x() * constraints[curColConst][j].data.x();
                        double w = n * n2 * compliance.getValue()[dof] ;
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
        sout<<" : "<<sendl;
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

template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::getComplianceMatrix(defaulttype::BaseMatrix *m)
{
    const VecReal &comp=compliance.getValue();
    const unsigned int s=comp.size();
    m->resize(s,s); //resize must set to zero the content of the matrix
    for (unsigned int l=0; l<s; ++l)
    {
        for (unsigned int c=0; c<s; ++c)
        {
            if (l==c) m->set(l,c,comp[l]);
        }
    }
}

template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::applyContactForce(const defaulttype::BaseVector *f)
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

/// in the Vec1Types and Vec3Types case, compliance is a vector of size mstate->getSize()
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



template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::setConstraintDForce(double * df, int begin, int end, bool update)
{
    /// set a force difference on a set of constraints (between constraint number "begin" and constraint number "end"
    /// if update is false, do nothing
    /// if update is true, it computes the displacements due to this delta of force.
    /// As the contact are uncoupled, a displacement is obtained only on dof involved with the constraints

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



template<class DataTypes>
void UncoupledConstraintCorrection<DataTypes>::getBlockDiagonalCompliance(defaulttype::BaseMatrix* W, int begin, int end)
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
                        double w = n1 * n2 * compliance.getValue()[dof1];
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
