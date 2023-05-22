/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/behavior/BaseConstraint.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <fstream>

namespace sofa::core::behavior
{

template<class DataTypes>
Mass<DataTypes>::Mass(MechanicalState<DataTypes> *mm)
    : ForceField<DataTypes>(mm)
    , m_gnuplotFileEnergy(nullptr)
{
}

template<class DataTypes>
Mass<DataTypes>::~Mass()
{
}

template<class DataTypes>
void Mass<DataTypes>::addMDx(const MechanicalParams* mparams, MultiVecDerivId fid, SReal factor)
{
    if (mparams)
    {
        auto mstate = this->mstate.get();
        addMDx(mparams, *fid[mstate].write(), *mparams->readDx(mstate), factor);
    }
}

template<class DataTypes>
void Mass<DataTypes>::addMDx(const MechanicalParams* /*mparams*/, DataVecDeriv& /*f*/, const DataVecDeriv& /*dx*/ , SReal /*factor*/ )
{
    msg_warning() << "Method addMDx(const MechanicalParams* , DataVecDeriv& , const DataVecDeriv&  , SReal  ) not implemented.";
}


template<class DataTypes>
void Mass<DataTypes>::accFromF(const MechanicalParams* mparams, MultiVecDerivId aid)
{
    if(mparams)
    {
        auto mstate = this->mstate.get();
        accFromF(mparams, *aid[mstate].write(), *mparams->readF(mstate));
    }
    else msg_error() <<"Mass<DataTypes>::accFromF(const MechanicalParams* mparams, MultiVecDerivId aid) receives no mparam";
}

template<class DataTypes>
void Mass<DataTypes>::accFromF(const MechanicalParams* /*mparams*/, DataVecDeriv& /*a*/, const DataVecDeriv& /*f*/)
{
    msg_warning() << "Method accFromF(const MechanicalParams* , DataVecDeriv& , const DataVecDeriv& ) not implemented.";
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
    sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());
#endif
}

template<class DataTypes>
void Mass<DataTypes>::addMBKdx(const MechanicalParams* mparams, MultiVecDerivId dfId)
{
    this->ForceField<DataTypes>::addMBKdx(mparams, dfId);
    if (mparams->mFactorIncludingRayleighDamping(rayleighMass.getValue()) != 0.0)
    {
        addMDx(mparams, *dfId[this->mstate.get()].write(),
                *mparams->readDx(this->mstate.get()), mparams->mFactorIncludingRayleighDamping(rayleighMass.getValue()));
    }
}

template<class DataTypes>
SReal Mass<DataTypes>::getKineticEnergy(const MechanicalParams* mparams) const
{
    if (this->mstate)
        return getKineticEnergy(mparams /* PARAMS FIRST */, *mparams->readV(this->mstate.get()));
    return 0.0;
}

template<class DataTypes>
SReal Mass<DataTypes>::getKineticEnergy(const MechanicalParams* /*mparams*/, const DataVecDeriv& /*v*/) const
{
    msg_warning() << "Method getKineticEnergy(const MechanicalParams*, const DataVecDeriv& ) not implemented.";
    return 0.0;
}


template<class DataTypes>
SReal Mass<DataTypes>::getPotentialEnergy(const MechanicalParams* mparams) const
{
    if (this->mstate)
        return getPotentialEnergy(mparams /* PARAMS FIRST */, *mparams->readX(this->mstate.get()));
    return 0.0;
}

template<class DataTypes>
SReal Mass<DataTypes>::getPotentialEnergy(const MechanicalParams* /*mparams*/, const DataVecCoord& /*x*/) const
{
    msg_warning() << "Method getPotentialEnergy( const MechanicalParams*, const DataVecCoord& ) not implemented.";
    return 0.0;
}


template<class DataTypes>
type::Vec6 Mass<DataTypes>::getMomentum( const MechanicalParams* mparams ) const
{
    auto state = this->mstate.get();
    if (state)
        return getMomentum(mparams, *mparams->readX(state), *mparams->readV(state));
    return type::Vec6();
}

template<class DataTypes>
type::Vec6 Mass<DataTypes>::getMomentum( const MechanicalParams* /*mparams*/, const DataVecCoord& /*x*/, const DataVecDeriv& /*v*/ ) const
{
    msg_warning() << "Method getMomentum( const MechanicalParams*, const DataVecCoord&, const DataVecDeriv& ) not implemented.";
    return type::Vec6();
}



template<class DataTypes>
void Mass<DataTypes>::addMToMatrix(const MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    if (r)
        addMToMatrix(r.matrix, mparams->mFactorIncludingRayleighDamping(rayleighMass.getValue()), r.offset);
}

template<class DataTypes>
void Mass<DataTypes>::addMToMatrix(sofa::linearalgebra::BaseMatrix * /*mat*/, SReal /*mFact*/, unsigned int &/*offset*/)
{
    static int i=0;
    if (i < 10) {
        msg_warning() << "Method addMToMatrix with Scalar not implemented";
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
void Mass<DataTypes>::addGravityToV(const MechanicalParams* mparams, MultiVecDerivId vid)
{
    if(this->mstate)
    {
        DataVecDeriv& v = *vid[this->mstate.get()].write();
        addGravityToV(mparams, v);
    }
}

template<class DataTypes>
void Mass<DataTypes>::addGravityToV(const MechanicalParams* /* mparams */, DataVecDeriv& /* d_v */)
{
    static int i=0;
    if (i < 10) {
        msg_warning() << "Method addGravityToV with Scalar not implemented";
        i++;
    }
}


template<class DataTypes>
void Mass<DataTypes>::initGnuplot(const std::string path)
{
    if (!this->getName().empty())
    {
        if (m_gnuplotFileEnergy != nullptr)
            delete m_gnuplotFileEnergy;

        m_gnuplotFileEnergy = new std::ofstream( (path+this->getName()+"_Energy.txt").c_str() );
    }
}

template<class DataTypes>
void Mass<DataTypes>::exportGnuplot(const MechanicalParams* mparams, SReal time)
{
    if (m_gnuplotFileEnergy!=nullptr)
    {
        (*m_gnuplotFileEnergy) << time <<"\t"<< this->getKineticEnergy(mparams)
                               <<"\t"<< this->getPotentialEnergy(mparams)
                              <<"\t"<< this->getPotentialEnergy(mparams)
                                +this->getKineticEnergy(mparams)<< std::endl;
    }
}

template <class DataTypes>
SReal Mass<DataTypes>::getElementMass(sofa::Index ) const
{
    msg_warning() << "Method getElementMass with Scalar not implemented";
    return 0.0;
}

template <class DataTypes>
void Mass<DataTypes>::getElementMass(sofa::Index, linearalgebra::BaseMatrix *m) const
{
    static const linearalgebra::BaseMatrix::Index dimension = (linearalgebra::BaseMatrix::Index) defaulttype::DataTypeInfo<Coord>::size();
    if (m->rowSize() != dimension || m->colSize() != dimension) m->resize(dimension,dimension);

    m->clear();
    msg_warning() << "Method getElementMass with Matrix not implemented";
}

} // namespace sofa::core::behavior
