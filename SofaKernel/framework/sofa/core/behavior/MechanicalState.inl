/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_BEHAVIOR_MECHANICALSTATE_INL_H
#define SOFA_CORE_BEHAVIOR_MECHANICALSTATE_INL_H

#include <sofa/helper/StringUtils.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/VecId.h>
#include <sofa/core/State.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include "MechanicalState.h"

namespace sofa
{

namespace core
{

namespace behavior
{

template<class TDataTypes>
size_t MechanicalState<TDataTypes>::getCoordDimension() const { return defaulttype::DataTypeInfo<Coord>::size(); }

template<class TDataTypes>
size_t MechanicalState<TDataTypes>::getDerivDimension() const { return defaulttype::DataTypeInfo<Deriv>::size(); }

template<class TDataTypes>
std::string MechanicalState<TDataTypes>::getTemplateName() const
{
    return templateName(this);
}

template<class TDataTypes>
std::string MechanicalState<TDataTypes>::templateName(const MechanicalState<DataTypes>*)
{
    return DataTypes::Name();
}

template<class TDataTypes>
template<class T>
std::string MechanicalState<TDataTypes>::shortName(const T* ptr, objectmodel::BaseObjectDescription* arg)
{
    std::string name = Inherit1::shortName(ptr, arg);
    sofa::helper::replaceAll(name, "Mechanical", "M");
    sofa::helper::replaceAll(name, "mechanical", "m");
    return name;
}

template<class TDataTypes>
void MechanicalState<TDataTypes>::copyToBuffer(SReal* dst, ConstVecId src, unsigned n) const {
    const size_t size = this->getSize();

    switch(src.type) {
    case V_COORD: {
        helper::ReadAccessor< Data<VecCoord> > vec = this->read(ConstVecCoordId(src));
        const size_t dim = defaulttype::DataTypeInfo<Coord>::size();
        assert( n == dim * size );

        for(size_t i = 0; i < size; ++i) {
            for(size_t j = 0; j < dim; ++j) {
                defaulttype::DataTypeInfo<Coord>::getValue(vec[i], j, *(dst++));
            }
        }

    }; break;
    case V_DERIV: {
        helper::ReadAccessor< Data<VecDeriv> > vec = this->read(ConstVecDerivId(src));
        const size_t dim = defaulttype::DataTypeInfo<Deriv>::size();
        assert( n == dim * size );

        for(size_t i = 0; i < size; ++i) {
            for(size_t j = 0; j < dim; ++j) {
                defaulttype::DataTypeInfo<Deriv>::getValue(vec[i], j, *(dst++));
            }
        }

    }; break;
    default:
        assert( false );
    }

    // get rid of unused parameter warnings in release build
    (void) n;
}

template<class TDataTypes>
void MechanicalState<TDataTypes>::copyFromBuffer(VecId dst, const SReal* src, unsigned n) {
    const size_t size = this->getSize();

    switch(dst.type) {
    case V_COORD: {
        helper::WriteOnlyAccessor< Data<VecCoord> > vec = this->write(VecCoordId(dst));
        const size_t dim = defaulttype::DataTypeInfo<Coord>::size();
        assert( n == dim * size );

        for(size_t i = 0; i < size; ++i) {
            for(size_t j = 0; j < dim; ++j) {
                defaulttype::DataTypeInfo<Coord>::setValue(vec[i], j, *(src++));
            }
        }

    }; break;
    case V_DERIV: {
        helper::WriteOnlyAccessor< Data<VecDeriv> > vec = this->write(VecDerivId(dst));
        const size_t dim = defaulttype::DataTypeInfo<Deriv>::size();
        assert( n == dim * size );

        for(size_t i = 0; i < size; ++i) {
            for(size_t j = 0; j < dim; ++j) {
                defaulttype::DataTypeInfo<Deriv>::setValue(vec[i], j, *(src++));
            }
        }

    }; break;
    default:
        assert( false );
    }

    // get rid of unused parameter warnings in release build
    (void) n;
}

template<class TDataTypes>
void MechanicalState<TDataTypes>::addFromBuffer(VecId dst, const SReal* src, unsigned n) {
    const size_t size = this->getSize();

    switch(dst.type) {
    case V_COORD: {
        helper::WriteAccessor< Data<VecCoord> > vec = this->write(VecCoordId(dst));
        const size_t dim = defaulttype::DataTypeInfo<Coord>::size();
        assert( n == dim * size );

        for(size_t i = 0; i < size; ++i) {
            for(size_t j = 0; j < dim; ++j) {
                typename Coord::value_type tmp;
                defaulttype::DataTypeInfo<Coord>::getValue(vec[i], j, tmp);
                tmp += (typename Coord::value_type) *(src++);
                defaulttype::DataTypeInfo<Coord>::setValue(vec[i], j, tmp);
            }
        }

    }; break;
    case V_DERIV: {
        helper::WriteAccessor< Data<VecDeriv> > vec = this->write(VecDerivId(dst));
        const size_t dim = defaulttype::DataTypeInfo<Deriv>::size();
        assert( n == dim * size );

        for(size_t i = 0; i < size; ++i) {
            for(size_t j = 0; j < dim; ++j) {
                typename Deriv::value_type tmp;
                defaulttype::DataTypeInfo<Deriv>::getValue(vec[i], j, tmp);
                tmp += (typename Coord::value_type) *(src++);
                defaulttype::DataTypeInfo<Deriv>::setValue(vec[i], j, tmp);
            }
        }

    }; break;
    default:
        assert( false );
    }

    // get rid of unused parameter warnings in release build
    (void) n;
}
} // namespace behavior

} // namespace core

} // namespace sofa

#endif
