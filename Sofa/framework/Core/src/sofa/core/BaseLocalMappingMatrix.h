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

#include <sofa/core/config.h>
#include <sofa/type/fwd.h>
#include <sofa/core/MatrixAccumulator.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/DerivativeMatrix.h>

namespace sofa::core
{
class GeometricStiffnessMatrix;

class SOFA_CORE_API MappingMatrixAccumulator : public MatrixAccumulatorInterface {};
class SOFA_CORE_API ListMappingMatrixAccumulator : public ListMatrixAccumulator<MappingMatrixAccumulator>{};

namespace matrixaccumulator
{

template<>
struct get_abstract_strong<Contribution::GEOMETRIC_STIFFNESS>
{
    using type = MappingMatrixAccumulator;
    using ComponentType = core::BaseMapping;
    using MatrixBuilderType = sofa::core::GeometricStiffnessMatrix;
};

template<>
struct get_list_abstract_strong<Contribution::GEOMETRIC_STIFFNESS>
{
    using type = ListMappingMatrixAccumulator;
    using ComponentType = core::BaseMapping;
};

}

class SOFA_CORE_API GeometricStiffnessMatrix
    : public DerivativeMatrix<matrixaccumulator::Contribution::GEOMETRIC_STIFFNESS>
{
public:

    struct DJ
    {
        DJ(BaseState* _mstate1, GeometricStiffnessMatrix* _mat)
            : mstate1(_mstate1), mat(_mat) {}

        Derivative withRespectToPositionsIn(BaseState* mstate2) const
        {
            return Derivative{this->mstate1, mstate2, this->mat};
        }

    private:
        BaseState* mstate1 { nullptr };
        GeometricStiffnessMatrix* mat { nullptr };
    };

    DJ getMappingDerivativeIn(BaseState* mstate)
    {
        return DJ{mstate, this};
    }
};

} //namespace sofa::core
