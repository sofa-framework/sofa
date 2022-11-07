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

#include <sofa/linearalgebra/EigenSparseMatrix.h>

SOFA_DISABLED_HEADER("v21.12", "v22.06", "sofa/linearalgebra/EigenSparseMatrix.h")
#include <SofaEigen2Solver/EigenBaseSparseMatrix.h>
#include <sofa/core/objectmodel/Data.h>

namespace sofa::component::linearsolver
{

    template<class InDataTypes, class OutDataTypes>
    struct EigenSparseMatrix : public sofa::linearalgebra::EigenSparseMatrix<InDataTypes, OutDataTypes>
    {
        typedef sofa::linearalgebra::EigenSparseMatrix<InDataTypes, OutDataTypes> Inherit;

        typedef typename InDataTypes::Deriv InDeriv;
        typedef typename InDataTypes::VecDeriv InVecDeriv;
        typedef typename InDataTypes::Real InReal;
        typedef typename OutDataTypes::Deriv OutDeriv;
        typedef typename OutDataTypes::VecDeriv OutVecDeriv;
        typedef typename OutDataTypes::Real OutReal;

        EigenSparseMatrix(Index nbRow=0, Index nbCol=0)
            :Inherit(nbRow, nbCol) {}

        using Inherit::mult;
        /// compute result = A * data
        void mult(Data<OutVecDeriv>& _result, const Data<InVecDeriv>& _data) const 
        {
            helper::WriteOnlyAccessor<Data<OutVecDeriv> > result(_result);
            helper::ReadAccessor<Data<InVecDeriv> > data(_data);

            this->mult_impl(result, data);
        }

        using Inherit::addMult;
        /// compute result += A * data
        void addMult(Data<OutVecDeriv>& result, const Data<InVecDeriv>& data) const 
        {
            helper::WriteAccessor<Data<OutVecDeriv> > res(result);
            helper::ReadAccessor<Data<InVecDeriv> > dat(data);

            this->addMult_impl(res, dat, 1.0);
        }

        /// compute result += A * data * fact
        void addMult(Data<OutVecDeriv>& result, const Data<InVecDeriv>& data, const OutReal fact) const 
        {
            helper::WriteAccessor<Data<OutVecDeriv> > res(result);
            helper::ReadAccessor<Data<InVecDeriv> > dat(data);

            this->addMult_impl(res, dat, fact);
        }

        using Inherit::addMultTranspose;
        /// compute result += A^T * data * fact
        void addMultTranspose(Data<InVecDeriv>& result, const Data<OutVecDeriv>& data, const OutReal fact) const 
        {
            helper::WriteAccessor<Data<InVecDeriv> > res(result);
            helper::ReadAccessor<Data<OutVecDeriv> > dat(data);

            this->addMultTranspose_impl(res, dat, fact);
        }

        /// compute result += A^T * data
        void addMultTranspose(Data<InVecDeriv>& result, const Data<OutVecDeriv>& data) const 
        {
            helper::WriteAccessor<Data<InVecDeriv> > res(result);
            helper::ReadAccessor<Data<OutVecDeriv> > dat(data);

            this->addMultTranspose_impl(res, dat, 1.0);
        }

    };


} // namespace sofa::component::linearsolver
