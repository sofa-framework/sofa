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

#include <sofa/core/State.h>
#include <sofa/core/topology/TopologyData.inl>

namespace sofa::core::visual
{

template< typename DataTypes >
class VisualState : public core::State< DataTypes >
{
public:
    SOFA_CLASS(VisualState, SOFA_TEMPLATE(core::State, defaulttype::Vec3Types));

    using VecCoord = typename DataTypes::VecCoord;
    using VecDeriv = typename DataTypes::VecCoord;
    using MatrixDeriv = typename DataTypes::MatrixDeriv;

    core::topology::PointData< VecCoord > m_positions; ///< Vertices coordinates
    core::topology::PointData< VecCoord > m_restPositions; ///< Vertices rest coordinates
    core::topology::PointData< VecDeriv > m_vnormals; ///< Normals of the model
    bool modified; ///< True if input vertices modified since last rendering

    VisualState();

    virtual void resize(Size vsize) override;
    virtual Size getSize() const override { return Size(m_positions.getValue().size()); }

    //State API
    virtual       Data<VecCoord>* write(core::VecCoordId  v) override;
    virtual const Data<VecCoord>* read(core::ConstVecCoordId  v)  const override;
    virtual Data<VecDeriv>* write(core::VecDerivId v) override;
    virtual const Data<VecDeriv>* read(core::ConstVecDerivId v) const override;

    virtual       Data<MatrixDeriv>* write(core::MatrixDerivId /* v */) override { return nullptr; }
    virtual const Data<MatrixDeriv>* read(core::ConstMatrixDerivId /* v */) const override { return nullptr; }
};

#if !defined(SOFA_CORE_VISUAL_VISUALSTATE_CPP)
extern template class SOFA_CORE_API VisualState< defaulttype::Vec3Types >;
#endif


} // namespace sofa::core::visual
