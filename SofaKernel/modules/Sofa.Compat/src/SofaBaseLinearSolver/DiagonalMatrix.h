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

#include <sofa/linearalgebra/DiagonalMatrix.h>

// SOFA_DEPRECATED_HEADER("v21.12", "v22.06", "sofa/linearalgebra/DiagonalMatrix.h")
// BlockDiagonalMatrix was located in DiagonalMatrix previously
#include <sofa/linearalgebra/BlockDiagonalMatrix.h>

namespace sofa::component::linearsolver
{

    template<typename T>
    using DiagonalMatrix = sofa::linearalgebra::DiagonalMatrix<T>;

    template<std::size_t LC, typename T = double>
    using BlockDiagonalMatrix = sofa::linearalgebra::BlockDiagonalMatrix<LC, T>;

} // namespace sofa::component::linearsolver
