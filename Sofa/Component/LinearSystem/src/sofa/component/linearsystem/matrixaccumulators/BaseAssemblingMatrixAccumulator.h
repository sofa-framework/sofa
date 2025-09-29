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

#include <sofa/core/BaseMatrixAccumulatorComponent.h>
#include <sofa/core/MatrixAccumulator.h>

namespace sofa::component::linearsystem
{

template<core::matrixaccumulator::Contribution c>
class BaseAssemblingMatrixAccumulator : public sofa::core::get_base_object_strong_type<c>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BaseAssemblingMatrixAccumulator, c), sofa::core::get_base_object_strong_type<c>);
    using ComponentType = typename Inherit1::ComponentType;

    using Inherit1::add;

    virtual void setMatrixSize(const sofa::type::Vec2u&);
    [[nodiscard]] sofa::type::Vec2u getMatrixSize() const;
    void setPositionInGlobalMatrix(const sofa::type::Vec2u&);
    void setFactor(SReal factor);
    void setGlobalMatrix(sofa::linearalgebra::BaseMatrix* globalMatrix);

    [[nodiscard]]
    type::Vec<2, unsigned> getPositionInGlobalMatrix() const;

    void addContributionsToMatrix(sofa::linearalgebra::BaseMatrix* /* globalMatrix */, SReal /* factor */, const sofa::type::Vec2u& /* positionInMatrix */) {}

protected:

    BaseAssemblingMatrixAccumulator();

    Data< sofa::type::Vec2u > d_matrixSize; ///< Size of the local matrix
    Data< sofa::type::Vec2u > d_positionInGlobalMatrix; ///< Position of this local matrix in the global matrix
    Data< SReal > d_factor; ///< Factor applied to matrix entries. This factor depends on the ODE solver and the associated component.

    sofa::linearalgebra::BaseMatrix* m_globalMatrix { nullptr };

    sofa::type::Vec<2, sofa::SignedIndex> m_cachedPositionInGlobalMatrix;
    SReal m_cachedFactor{};
};

template<core::matrixaccumulator::Contribution c>
BaseAssemblingMatrixAccumulator<c>::BaseAssemblingMatrixAccumulator()
: Inherit1()
, d_matrixSize            (initData(&d_matrixSize            , sofa::type::Vec2u{}   , "matrixSize"            , "Size of the local matrix"))
, d_positionInGlobalMatrix(initData(&d_positionInGlobalMatrix, sofa::type::Vec2u{}   , "positionInGlobalMatrix", "Position of this local matrix in the global matrix"))
, d_factor                (initData(&d_factor                , static_cast<SReal>(1.), "factor"                , "Factor applied to matrix entries. This factor depends on the ODE solver and the associated component."))
{
    d_matrixSize.setReadOnly(true);
    d_positionInGlobalMatrix.setReadOnly(true);
    d_factor.setReadOnly(true);
}


template<core::matrixaccumulator::Contribution c>
void BaseAssemblingMatrixAccumulator<c>::setMatrixSize(const sofa::type::Vec2u& matrixSize)
{
    d_matrixSize.setValue(matrixSize);
}

template <core::matrixaccumulator::Contribution c>
sofa::type::Vec2u BaseAssemblingMatrixAccumulator<c>::getMatrixSize() const
{
    return d_matrixSize.getValue();
}

template <core::matrixaccumulator::Contribution c>
void BaseAssemblingMatrixAccumulator<c>::setPositionInGlobalMatrix(const sofa::type::Vec2u& pos)
{
    d_positionInGlobalMatrix.setValue(pos);
    m_cachedPositionInGlobalMatrix = type::toVecN<2, sofa::SignedIndex>(pos); // vec2u -> vec2i
}

template<core::matrixaccumulator::Contribution c>
void BaseAssemblingMatrixAccumulator<c>::setFactor(SReal factor)
{
    d_factor.setValue(factor);
    m_cachedFactor = factor;
}

template<core::matrixaccumulator::Contribution c>
void BaseAssemblingMatrixAccumulator<c>::setGlobalMatrix(sofa::linearalgebra::BaseMatrix* globalMatrix)
{
    m_globalMatrix = globalMatrix;
}

template <core::matrixaccumulator::Contribution c>
type::Vec<2, unsigned> BaseAssemblingMatrixAccumulator<c>::getPositionInGlobalMatrix() const
{
    return d_positionInGlobalMatrix.getValue();
}

}
