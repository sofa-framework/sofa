﻿/******************************************************************************
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
#define SOFA_COMPONENT_LINEARSOLVER_DIRECT_EIGENSPARSEQR_CPP
#include <sofa/component/linearsolver/direct/EigenSparseQR.h>
#include <sofa/component/linearsolver/direct/EigenDirectSparseSolver.inl>

#include <sofa/core/ObjectFactory.h>

namespace sofa::component::linearsolver::direct
{

template class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API EigenSparseQR< SReal >;
template class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API EigenSparseQR< sofa::type::Mat<3,3,SReal>>;

void registerEigenSparseQR(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Direct linear solver using a Sparse QR factorization.")
        .add< EigenSparseQR< SReal > >()
        .add< EigenSparseQR< sofa::type::Mat<3, 3, SReal> > >());
}

}
