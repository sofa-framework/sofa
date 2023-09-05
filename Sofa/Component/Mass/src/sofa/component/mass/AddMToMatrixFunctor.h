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
#include <sofa/component/mass/config.h>

#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::mass
{

/**
 * Helper struct to add entries in a matrix, based on the type of Mass (MassType).
 *
 * This class is specialized for Rigid types.
 *
 * The default implementation assumes it deals with Vec types: Deriv is a Vec type, and
 * MassType is a floating point.
 */
template<class Deriv, class MassType, class MatrixType>
struct AddMToMatrixFunctor
{
    static_assert(std::is_floating_point_v<MassType>, "Default implementation of AddMToMatrixFunctor assumes MassType is a floating point");

    void operator()(MatrixType * mat, MassType mass, int pos, MassType fact)
    {
        this->operator()(mat, mass, pos, pos, fact);
    }

    ///Method to add non-diagonal terms
    void operator()(MatrixType * mat, MassType mass, int posRow, int posColumn, MassType fact)
    {
        const auto m = mass * fact;
        for (unsigned int i = 0; i < Deriv::total_size; ++i)
            mat->add(posRow + i, posColumn + i, m);
    }
};

/**
 * Specialization for Rigid types
 */
template<sofa::Size N, typename Real, class MatrixType>
struct AddMToMatrixFunctor< defaulttype::RigidDeriv<N,Real>, defaulttype::RigidMass<N,Real>, MatrixType >
{
    void operator()(MatrixType * mat, const defaulttype::RigidMass<N,Real>& mass, int pos, Real fact)
    {
        const auto m = mass.mass * fact;
        for (sofa::Size i = 0; i < N; ++i)
            mat->add(pos + i, pos + i, m);
        mat->add(pos + N, pos + N, mass.inertiaMassMatrix * fact);
    }
};

} // namespace sofa::component::mass
