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
#include <sofa/component/linearsolver/direct/config.h>

#include <sofa/component/linearsolver/direct/EigenDirectSparseSolver.h>

namespace sofa::component::linearsolver::direct
{

/**
 * Linear solver based on direct sparse supernodal LU factorization
 *
 * The factorization is based on the Eigen library
 */
template<class TBlockType>
class EigenSparseLU
    : public EigenDirectSparseSolver<
        TBlockType,
        MainLUFactory
    >
{
public:
    typedef sofa::linearalgebra::CompressedRowSparseMatrix<TBlockType> Matrix;
    using Real = typename Matrix::Real;
    typedef sofa::linearalgebra::FullVector<Real> Vector;

    SOFA_CLASS(SOFA_TEMPLATE(EigenSparseLU, TBlockType), SOFA_TEMPLATE2(EigenDirectSparseSolver, TBlockType, MainLUFactory));

    bool supportNonSymmetricSystem() const override { return true; }
};

#ifndef SOFA_COMPONENT_LINEARSOLVER_DIRECT_EIGENSPARSELU_CPP
    extern template class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API EigenSparseLU< SReal >;
    extern template class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API EigenSparseLU< sofa::type::Mat<3,3,SReal> >;
#endif

}
