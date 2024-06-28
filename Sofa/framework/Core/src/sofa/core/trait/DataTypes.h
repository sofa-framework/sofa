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

template <typename T>
using Coord_t = typename T::Coord;

template <typename T>
using Real_t = typename T::Real;

template <typename T>
using VecReal_t = typename T::VecReal;

template <typename T>
using Deriv_t = typename T::Deriv;

template <typename T>
using MatrixDeriv_t = typename T::MatrixDeriv;

template <typename T>
using VecCoord_t = typename T::VecCoord;

template <typename T>
using VecDeriv_t = typename T::VecDeriv;

namespace core::objectmodel
{
template<typename T>
class Data;
}

template <typename T>
using DataVecCoord_t = core::objectmodel::Data<VecCoord_t<T>>;

template <typename T>
using DataVecDeriv_t = core::objectmodel::Data<VecDeriv_t<T>>;

template <typename T>
using DataMatrixDeriv_t = core::objectmodel::Data<MatrixDeriv_t<T>>;

}
