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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_BEHAVIOR_MASS_INL
#define SOFA_CORE_BEHAVIOR_MASS_INL

#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/BaseConstraint.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/defaulttype/DataTypeInfo.h>

namespace sofa
{

namespace core
{

namespace behavior
{

template<class DataTypes>
Mass<DataTypes>::Mass(MechanicalState<DataTypes> *mm)
    : ForceField<DataTypes>(mm),m_gnuplotFileEnergy(NULL)
{
}

template<class DataTypes>
Mass<DataTypes>::~Mass()
{
}

#ifndef SOFA_SMP
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

#endif
template<class DataTypes>
double Mass<DataTypes>::getKineticEnergy() const
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
void Mass<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix * /*mat*/, double /*kFact*/, unsigned int &/*offset*/)
{
}

template<class DataTypes>
void Mass<DataTypes>::addMToMatrix(sofa::defaulttype::BaseMatrix * /*mat*/, double /*mFact*/, unsigned int &/*offset*/)
{
    serr << "addMToMatrix not implemented by " << this->getClassName() << sendl;
}

template<class DataTypes>
void Mass<DataTypes>::addMBKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, double mFact, double bFact, double kFact)
{
    this->ForceField<DataTypes>::addMBKToMatrix(matrix, mFact, bFact, kFact);
    if (mFact != 0.0)
        addMToMatrix(matrix, mFact);
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

#ifdef SOFA_SMP
template<class DataTypes>
struct ParallelMassAccFromF
{
    void	operator()( Mass< DataTypes >*m,Shared_rw< typename  DataTypes::VecDeriv> _a,Shared_r< typename  DataTypes::VecDeriv> _f)
    {
        m->accFromF(_a.access(),_f.read());
    }
};
template<class DataTypes>
struct ParallelMassAddMDx
{
public:
    void	operator()(Mass< DataTypes >*m,Shared_rw< typename DataTypes::VecDeriv> _res,Shared_r<  typename DataTypes::VecDeriv> _dx,double factor)
    {
        m->addMDx(_res.access(),_dx.read(),factor);
    }
};

template<class DataTypes>
struct ParallelMassAddMDxCPU
{
public:
    void	operator()( Mass< DataTypes >* m,Shared_rw< typename DataTypes::VecDeriv> _res,Shared_r<  typename DataTypes::VecDeriv> _dx,double factor)
    {
        m->addMDx(_res.access(),_dx.read(),factor);
    }
};

template<class DataTypes>
struct ParallelMassAccFromFCPU
{
    void	operator()( Mass< DataTypes >*m,Shared_rw< typename  DataTypes::VecDeriv> _a,Shared_r< typename  DataTypes::VecDeriv> _f)
    {
        m->accFromF(_a.access(),_f.read());
    }
};


template<class DataTypes>
void Mass< DataTypes >::addMDx( double factor)
{
    Task<ParallelMassAddMDxCPU < DataTypes  > ,ParallelMassAddMDx < DataTypes > >(this,**this->mstate->getF(), **this->mstate->getDx(),factor);
}

template<class DataTypes>
void Mass< DataTypes >::accFromF()
{
    Task<ParallelMassAccFromFCPU< DataTypes  >,ParallelMassAccFromF< DataTypes  > >(this,**this->mstate->getDx(), **this->mstate->getF());
}

#endif

/// return the mass relative to the DOF #index
template <class DataTypes>
double Mass<DataTypes>::getElementMass(unsigned int ) const
{
    serr<<"getElementMass with Scalar not implemented"<<sendl;
    return 0.0;
}
//TODO: special case for Rigid Mass
template <class DataTypes>
void Mass<DataTypes>::getElementMass(unsigned int , defaulttype::BaseMatrix *m) const
{
    static unsigned int dimension = defaulttype::DataTypeInfo<Coord>::size();
    if (m->rowSize() != dimension || m->colSize() != dimension) m->resize(dimension,dimension);

    m->clear();
    serr<<"getElementMass with Matrix not implemented"<<sendl;
}



} // namespace behavior

} // namespace core

} // namespace sofa

#endif
