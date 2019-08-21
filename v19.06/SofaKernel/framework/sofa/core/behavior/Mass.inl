/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_BEHAVIOR_MASS_INL
#define SOFA_CORE_BEHAVIOR_MASS_INL

#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/BaseConstraint.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <fstream>


namespace sofa
{

namespace core
{

namespace behavior
{

template<class DataTypes>
Mass<DataTypes>::Mass(MechanicalState<DataTypes> *mm)
    : ForceField<DataTypes>(mm)
    , m_gnuplotFileEnergy(NULL)
{
}

template<class DataTypes>
Mass<DataTypes>::~Mass()
{
}

template<class DataTypes>
void Mass<DataTypes>::init()
{
    ForceField<DataTypes>::init();
}


template<class DataTypes>
void Mass<DataTypes>::addMDx(const MechanicalParams* mparams, MultiVecDerivId fid, SReal factor)
{
    if (mparams)
    {
            addMDx(mparams, *fid[this->mstate.get(mparams)].write(), *mparams->readDx(this->mstate), factor);
    }
}

template<class DataTypes>
void Mass<DataTypes>::addMDx(const MechanicalParams* /*mparams*/, DataVecDeriv& /*f*/, const DataVecDeriv& /*dx*/ , SReal /*factor*/ )
{
    msg_error()<< getClassName()<< ": addMDx(const MechanicalParams* , DataVecDeriv& , const DataVecDeriv&  , SReal  ) not implemented.";
}


template<class DataTypes>
void Mass<DataTypes>::accFromF(const MechanicalParams* mparams, MultiVecDerivId aid)
{
    if(mparams)
    {
            accFromF(mparams, *aid[this->mstate.get(mparams)].write(), *mparams->readF(this->mstate));
    }
    else msg_error() <<"Mass<DataTypes>::accFromF(const MechanicalParams* mparams, MultiVecDerivId aid) receives no mparam";
}

template<class DataTypes>
void Mass<DataTypes>::accFromF(const MechanicalParams* /*mparams*/, DataVecDeriv& /*a*/, const DataVecDeriv& /*f*/)
{
    msg_error()<<getClassName()<<": accFromF(const MechanicalParams* , DataVecDeriv& , const DataVecDeriv& ) not implemented.";
}


template<class DataTypes>
void Mass<DataTypes>::addDForce(const MechanicalParams*
                                #ifndef NDEBUG
                                mparams
                                #endif
                                ,
                                DataVecDeriv & /*df*/, const DataVecDeriv & /*dx*/)
{
#ifndef NDEBUG
    // @TODO Remove
    // Hack to disable warning message
    mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
#endif
}

template<class DataTypes>
void Mass<DataTypes>::addMBKdx(const MechanicalParams* mparams, MultiVecDerivId dfId)
{
    this->ForceField<DataTypes>::addMBKdx(mparams, dfId);
    if (mparams->mFactorIncludingRayleighDamping(rayleighMass.getValue()) != 0.0)
    {
        addMDx(mparams, *dfId[this->mstate.get(mparams)].write(), *mparams->readDx(this->mstate), mparams->mFactorIncludingRayleighDamping(rayleighMass.getValue()));
    }
}

template<class DataTypes>
SReal Mass<DataTypes>::getKineticEnergy(const MechanicalParams* mparams) const
{
    if (this->mstate)
        return getKineticEnergy(mparams /* PARAMS FIRST */, *mparams->readV(this->mstate));
    return 0.0;
}

template<class DataTypes>
SReal Mass<DataTypes>::getKineticEnergy(const MechanicalParams* /*mparams*/, const DataVecDeriv& /*v*/) const
{
    msg_error() <<getClassName()<<": getKineticEnergy(const MechanicalParams*, const DataVecDeriv& ) not implemented.";
    return 0.0;
}


template<class DataTypes>
SReal Mass<DataTypes>::getPotentialEnergy(const MechanicalParams* mparams) const
{
    if (this->mstate)
        return getPotentialEnergy(mparams /* PARAMS FIRST */, *mparams->readX(this->mstate));
    return 0.0;
}

template<class DataTypes>
SReal Mass<DataTypes>::getPotentialEnergy(const MechanicalParams* /*mparams*/, const DataVecCoord& /*x*/) const
{
    msg_error() << getClassName()<<": getPotentialEnergy( const MechanicalParams*, const DataVecCoord& ) not implemented.";
    return 0.0;
}


template<class DataTypes>
defaulttype::Vector6 Mass<DataTypes>::getMomentum( const MechanicalParams* mparams ) const
{
    if (this->mstate)
        return getMomentum(mparams, *mparams->readX(this->mstate), *mparams->readV(this->mstate));
    return defaulttype::Vector6();
}

template<class DataTypes>
defaulttype::Vector6 Mass<DataTypes>::getMomentum( const MechanicalParams* /*mparams*/, const DataVecCoord& /*x*/, const DataVecDeriv& /*v*/ ) const
{
    msg_error() << getClassName()<<": getMomentum( const MechanicalParams*, const DataVecCoord&, const DataVecDeriv& ) not implemented.";
    return defaulttype::Vector6();
}



template<class DataTypes>
void Mass<DataTypes>::addMToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    if (r)
        addMToMatrix(r.matrix, mparams->mFactorIncludingRayleighDamping(rayleighMass.getValue()), r.offset);
}

template<class DataTypes>
void Mass<DataTypes>::addMToMatrix(sofa::defaulttype::BaseMatrix * /*mat*/, SReal /*mFact*/, unsigned int &/*offset*/)
{
    static int i=0;
    if (i < 10) {
        msg_error() <<getClassName()<<": addMToMatrix not implemented.";
        i++;
    }
}

template<class DataTypes>
void Mass<DataTypes>::addMBKToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    this->ForceField<DataTypes>::addMBKToMatrix(mparams, matrix);
    if (mparams->mFactorIncludingRayleighDamping(rayleighMass.getValue()) != 0.0)
        addMToMatrix(mparams, matrix);
}

template<class DataTypes>
void Mass<DataTypes>::addSubMBKToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix, const helper::vector<unsigned> /*subMatrixIndex*/) {
    addMBKToMatrix(mparams,matrix); // default implementation use full addMFunction
}

template<class DataTypes>
void Mass<DataTypes>::addGravityToV(const MechanicalParams* mparams, MultiVecDerivId vid)
{
    if(this->mstate)
    {
        DataVecDeriv& v = *vid[this->mstate.get(mparams)].write();
        addGravityToV(mparams, v);
    }
}

template<class DataTypes>
void Mass<DataTypes>::addGravityToV(const MechanicalParams* /* mparams */, DataVecDeriv& /* d_v */)
{
    static int i=0;
    if (i < 10) {
        msg_error() <<getClassName()<<": addGravityToV not implemented.";
        i++;
    }
}


template<class DataTypes>
void Mass<DataTypes>::initGnuplot(const std::string path)
{
    if (!this->getName().empty())
    {
        if (m_gnuplotFileEnergy != NULL)
            delete m_gnuplotFileEnergy;

        m_gnuplotFileEnergy = new std::ofstream( (path+this->getName()+"_Energy.txt").c_str() );
    }
}

template<class DataTypes>
void Mass<DataTypes>::exportGnuplot(const MechanicalParams* mparams, SReal time)
{
    if (m_gnuplotFileEnergy!=NULL)
    {
        (*m_gnuplotFileEnergy) << time <<"\t"<< this->getKineticEnergy(mparams)
                               <<"\t"<< this->getPotentialEnergy(mparams)
                              <<"\t"<< this->getPotentialEnergy(mparams)
                                +this->getKineticEnergy(mparams)<< std::endl;
    }
}

template <class DataTypes>
SReal Mass<DataTypes>::getElementMass(unsigned int ) const
{
    msg_error() <<getClassName()<<": getElementMass with Scalar not implemented";
    return 0.0;
}

template <class DataTypes>
void Mass<DataTypes>::getElementMass(unsigned int , defaulttype::BaseMatrix *m) const
{
    static const defaulttype::BaseMatrix::Index dimension = (defaulttype::BaseMatrix::Index) defaulttype::DataTypeInfo<Coord>::size();
    if (m->rowSize() != dimension || m->colSize() != dimension) m->resize(dimension,dimension);

    m->clear();
     msg_error()<<getClassName()<<": getElementMass with Matrix not implemented";
}

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
