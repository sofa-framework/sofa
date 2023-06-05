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
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/State.inl>

namespace sofa::core::behavior
{
template<class T>
void MechanicalState<T>::copyToBuffer(SReal* dst, ConstVecId src, unsigned n) const
{
    const auto size = this->getSize();

    switch(src.type) {
    case V_COORD: {
        sofa::helper::ReadAccessor< Data<VecCoord> > vec = this->read(ConstVecCoordId(src));
        const auto dim = defaulttype::DataTypeInfo<Coord>::size();
        assert( n == dim * size );

        for(Size i = 0; i < size; ++i) {
            for(Size j = 0; j < dim; ++j) {
                defaulttype::DataTypeInfo<Coord>::getValue(vec[i], j, *(dst++));
            }
        }

    }; break;
    case V_DERIV: {
        helper::ReadAccessor< Data<VecDeriv> > vec = this->read(ConstVecDerivId(src));
        const auto dim = defaulttype::DataTypeInfo<Deriv>::size();
        assert( n == dim * size );

        for(Size i = 0; i < size; ++i) {
            for(Size j = 0; j < dim; ++j) {
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

template<class T>
void MechanicalState<T>::copyFromBuffer(VecId dst, const SReal* src, unsigned n)
{
    const auto size = this->getSize();

    switch(dst.type) {
    case V_COORD: {
        helper::WriteOnlyAccessor< Data<VecCoord> > vec = this->write(VecCoordId(dst));
        const auto dim = defaulttype::DataTypeInfo<Coord>::size();
        assert( n == dim * size );

        for(Size i = 0; i < size; ++i) {
            for(Size j = 0; j < dim; ++j) {
                defaulttype::DataTypeInfo<Coord>::setValue(vec[i], j, *(src++));
            }
        }

    }; break;
    case V_DERIV: {
        helper::WriteOnlyAccessor< Data<VecDeriv> > vec = this->write(VecDerivId(dst));
        const auto dim = defaulttype::DataTypeInfo<Deriv>::size();
        assert( n == dim * size );

        for(Size i = 0; i < size; ++i) {
            for(Size j = 0; j < dim; ++j) {
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

template<class T>
void MechanicalState<T>::addFromBuffer(VecId dst, const SReal* src, unsigned n)
{
    const auto size = this->getSize();

    switch(dst.type) {
    case V_COORD: {
        helper::WriteAccessor< Data<VecCoord> > vec = this->write(VecCoordId(dst));
        const auto dim = defaulttype::DataTypeInfo<Coord>::size();
        assert( n == dim * size );

        for(Size i = 0; i < size; ++i) {
            for(Size j = 0; j < dim; ++j) {
                typename Coord::value_type tmp;
                defaulttype::DataTypeInfo<Coord>::getValue(vec[i], j, tmp);
                tmp += (typename Coord::value_type) *(src++);
                defaulttype::DataTypeInfo<Coord>::setValue(vec[i], j, tmp);
            }
        }

    }; break;
    case V_DERIV: {
        helper::WriteAccessor< Data<VecDeriv> > vec = this->write(VecDerivId(dst));
        const auto dim = defaulttype::DataTypeInfo<Deriv>::size();
        assert( n == dim * size );

        for(Size i = 0; i < size; ++i) {
            for(Size j = 0; j < dim; ++j) {
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
}
