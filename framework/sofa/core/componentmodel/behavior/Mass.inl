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
        const sofa::helper::vector< unsigned int >  offset,
        const defaulttype::BaseVector &FixedPoints)
{

    const unsigned int dimension=defaulttype::DataTypeInfo< Deriv >::size();
    const unsigned int numDofs=this->mstate->getSize();
    const unsigned int numConstraints=A.rowSize();

    if (invM_Jtrans.rowSize()*invM_Jtrans.colSize() == 0)
        invM_Jtrans.resize(numDofs*dimension,A.rowSize());

    const VecConst& unordered_c = *this->mstate->getC();

    //Pre-process the Vec Const to order them, and remove duplication
    sofa::helper::vector< std::map< unsigned int, Deriv > > c; c.resize(numConstraints);
    for (unsigned int system=0; system<constraintId.size(); ++system)
    {
        for (unsigned int i=0; i<constraintId[system].size(); ++i)
        {
            std::map< unsigned int, Deriv > &orderedConstraint=c[i+offset[system]];
            //typename std::map< unsigned int, Deriv >::iterator it;

            const SparseVecDeriv &V=unordered_c[ constraintId[system][i] ];
            for (unsigned int j=0; j<V.size(); ++j)
            {
                unsigned int dof=V[j].index;
                orderedConstraint[ dof ]+=V[j].data*factor[system];
            }
        }
    }



//     for (unsigned int n1=0;n1<c.size();++n1)
//       {
//      typename std::map< unsigned int, Deriv >::iterator itdebug;
//      sout << "Constraint " << n1 << ":" << sendl;
//      for (itdebug=c[n1].begin();itdebug!=c[n1].end();itdebug++)
//        {
//          sout << "\t[Dof=" << itdebug->first << ",Direction=" << itdebug->second << "] " << sendl;
//        }
//      sout << "" << sendl;
//       }
    //In c, we have ordered the contraints.
    //Filling the sparse matrices
    for (unsigned int n1=0; n1<c.size(); ++n1)
    {
        typename std::map< unsigned int, Deriv >::iterator it1;
        for (it1=c[n1].begin(); it1!=c[n1].end(); it1++)
        {
            unsigned int dof=it1->first;
            double invMassElement=1.0/this->getElementMass(dof);
            //Building M^-1.J^T
            Deriv v=it1->second;
            for (unsigned int d=0; d<dimension; ++d)
            {
                invM_Jtrans.add(dimension*dof+d,
                        n1,
                        v[d]*FixedPoints.element(dof)*invMassElement);
            }

            //Accumulating A=J.M^-1.J^T
            for (unsigned int n2=0; n2<c.size(); ++n2)
            {
                A.add(n1,n2,c[n1][dof]*c[n2][dof]*FixedPoints.element(dof)*invMassElement);
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
