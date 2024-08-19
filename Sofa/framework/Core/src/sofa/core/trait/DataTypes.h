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

namespace sofa
{

template <typename DataTypes>
using Coord_t = typename DataTypes::Coord;

template <typename DataTypes>
using Real_t = typename DataTypes::Real;

template <typename DataTypes>
using VecReal_t = typename DataTypes::VecReal;

template <typename DataTypes>
using Deriv_t = typename DataTypes::Deriv;

template <typename DataTypes>
using MatrixDeriv_t = typename DataTypes::MatrixDeriv;

template <typename DataTypes>
using VecCoord_t = typename DataTypes::VecCoord;

template <typename DataTypes>
using VecDeriv_t = typename DataTypes::VecDeriv;

namespace core::objectmodel
{
template<typename DataTypes>
class Data;
}

template <typename DataTypes>
using DataVecCoord_t = core::objectmodel::Data<VecCoord_t<DataTypes>>;

template <typename DataTypes>
using DataVecDeriv_t = core::objectmodel::Data<VecDeriv_t<DataTypes>>;

template <typename DataTypes>
using DataMatrixDeriv_t = core::objectmodel::Data<MatrixDeriv_t<DataTypes>>;

}
