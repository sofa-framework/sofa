/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_MASS_INL
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_MASS_INL

#include <sofa/core/componentmodel/behavior/Mass.h>
#include <sofa/core/componentmodel/behavior/BaseConstraint.h>
#include <sofa/core/componentmodel/behavior/ForceField.inl>
#include <sofa/defaulttype/DataTypeInfo.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

template<class DataTypes>
Mass<DataTypes>::Mass(MechanicalState<DataTypes> *mm)
    : ForceField<DataTypes>(mm),m_gnuplotFileEnergy(NULL)
{
    m_separateGravity = initData(&m_separateGravity , false, "separateGravity", "add separately gravity to velocity computation") ;
}

template<class DataTypes>
Mass<DataTypes>::~Mass()
{
}

template<class DataTypes>
void Mass<DataTypes>::addMDx(double factor)
{
    if (this->mstate)
        addMDx(*this->mstate->getF(), *this->mstate->getDx(), factor);
}

template<class DataTypes>
void Mass<DataTypes>::accFromF()
{
    if (this->mstate)
        accFromF(*this->mstate->getDx(), *this->mstate->getF());
}

template<class DataTypes>
double Mass<DataTypes>::getKineticEnergy()
{
    if (this->mstate)
        return getKineticEnergy(*this->mstate->getV());
    return 0;
}

template<class DataTypes>
void Mass<DataTypes>::addMBKdx(double mFactor, double bFactor, double kFactor)
{
    this->ForceField<DataTypes>::addMBKdx(mFactor, bFactor, kFactor);
    if (mFactor != 0.0)
        addMDx(mFactor);
}

template<class DataTypes>
void Mass<DataTypes>::addMBKv(double mFactor, double bFactor, double kFactor)
{
    this->ForceField<DataTypes>::addMBKv(mFactor, bFactor, kFactor);
    if (mFactor != 0.0)
    {
        if (this->mstate)
            addMDx(*this->mstate->getF(), *this->mstate->getV(), mFactor);
    }
}

template<class DataTypes>
void Mass<DataTypes>::addMBKToMatrix(sofa::defaulttype::BaseMatrix * matrix, double mFact, double bFact, double kFact, unsigned int &offset)
{
    this->ForceField<DataTypes>::addMBKToMatrix(matrix, mFact, bFact, kFact, offset);
    if (mFact != 0.0)
        addMToMatrix(matrix, mFact, offset);
}

template<class DataTypes>
void Mass<DataTypes>::initGnuplot(const std::string path)
{
    if( !this->getName().empty() )
    {

        if (m_gnuplotFileEnergy != NULL) delete m_gnuplotFileEnergy;
        m_gnuplotFileEnergy = new std::ofstream( (path+this->getName()+"_Energy.txt").c_str() );

    }
}

template<class DataTypes>
void Mass<DataTypes>::exportGnuplot(double time)
{
    if( m_gnuplotFileEnergy!=NULL )
    {
        (*m_gnuplotFileEnergy) << time <<"\t"<< this->getKineticEnergy() <<"\t"<< this->getPotentialEnergy() <<"\t"<< this->getPotentialEnergy()+this->getKineticEnergy()<< sendl;
    }
}



template<class DataTypes>
void Mass<DataTypes>::buildSystemMatrix(defaulttype::BaseMatrix &invM_Jtrans , defaulttype::BaseMatrix & A ,
        const sofa::helper::vector< sofa::helper::vector<unsigned int>  >& constraintId ,
        const sofa::helper::vector< double >  factor ,
        const sofa::helper::vector< unsigned int > offset,
        const defaulttype::BaseVector &FixedPoints)
{

    //constraintId is a vector of vector, as the first dimension corresponds to the cosntraint components, and the second, the indices of the line of the VecConst
    const unsigned int dimension=defaulttype::DataTypeInfo< Deriv >::size();
    const unsigned int numDofs=this->mstate->getSize();
    const VecConst& c = *this->mstate->getC();


    if (invM_Jtrans.rowSize()*invM_Jtrans.colSize() == 0)
        invM_Jtrans.resize(numDofs*dimension,A.rowSize());

    typedef typename std::map< unsigned int, Deriv>::const_iterator constraintIterator;
    //Filling the sparse matrices
    //For all the constraint components found in the scene graph
    for (unsigned int system=0; system<constraintId.size(); ++system)
    {
        //For all the equations written by these constraints
        for (unsigned int n1=0; n1<constraintId[system].size(); ++n1)
        {
            //We read the non null entries of the SparseVector
            SparseVecDeriv sc1 = c[constraintId[system][n1]];
            for (constraintIterator it1=sc1.getData().begin(); it1!=sc1.getData().end(); it1++)
            {
                unsigned int dof=it1->first;
                Deriv v=it1->second;
                double invMassElement=1.0/this->getElementMass(dof);


                //Building M^-1.J^T
                //For the moment, M is considered as diagonal. Null term corresponds to FixedPoints
                for (unsigned int d=0; d<dimension; ++d)
                {
                    invM_Jtrans.add(dof*dimension+d,
                            offset[system]+n1,
                            v[d]*factor[system]*FixedPoints.element(dof)*invMassElement);
                }

                //Accumulating A=J.M^-1.J^T
                for (unsigned int system2=0; system2<constraintId.size(); ++system2)
                {
                    for (unsigned int n2=0; n2<constraintId[system2].size(); ++n2)
                    {
                        SparseVecDeriv sc2 = c[constraintId[system2][n2]];
                        SReal value=sc1.getDataAt(dof)*sc2.getDataAt(dof)*factor[system]*factor[system2]*FixedPoints.element(dof)*invMassElement;
                        A.add(n1+offset[system],n2+offset[system2],value);
                    }
                }
            }
        }
    }
}

template<class DataTypes>
void Mass<DataTypes>::buildInvMassDenseMatrix(defaulttype::BaseMatrix &m)
{
    unsigned int dimension=defaulttype::DataTypeInfo< Coord >::size();
    unsigned int numDofs=this->mstate->getSize();
    m.resize(dimension*numDofs,dimension*numDofs);
    for (unsigned int i=0; i<numDofs; ++i)
    {
        for (unsigned int j=0; j<dimension; ++j)
        {
            m.set(i*dimension+j,i*dimension+j,1.0/this->getElementMass(i));
        }
    }
}

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
