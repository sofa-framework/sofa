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
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_MASS_INL
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_MASS_INL

#include <sofa/core/componentmodel/behavior/Mass.h>
#include <sofa/core/componentmodel/behavior/ForceField.inl>

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
        (*m_gnuplotFileEnergy) << time <<"\t"<< this->getKineticEnergy() <<"\t"<< this->getPotentialEnergy() <<"\t"<< this->getPotentialEnergy()+this->getKineticEnergy()<< std::endl;
    }
}


} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
