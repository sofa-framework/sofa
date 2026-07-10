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
#include <sofa/component/engine/generate/config.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/behavior/SingleStateAccessor.h>
#include <sofa/core/behavior/TopologyAccessor.h>

namespace sofa::component::engine::generate
{

template<class DataTypes>
class LinearToHigherOrderElements :
    public sofa::core::DataEngine,
    public sofa::core::behavior::TopologyAccessor,
    public sofa::core::behavior::SingleStateAccessor<DataTypes>
{
public:
    SOFA_CLASS3(LinearToHigherOrderElements<DataTypes>,
        sofa::core::DataEngine,
        sofa::core::behavior::TopologyAccessor,
        sofa::core::behavior::SingleStateAccessor<DataTypes>);

    template<class ElementType> //e.g. sofa::geometry::Edge
    using TopologyElement = sofa::topology::Element<ElementType>;

    template<class ElementType> //e.g. sofa::geometry::Edge
    using SeqElement = sofa::type::vector<TopologyElement<ElementType>>;

    void init() override;

    sofa::DataVecCoord_t<DataTypes> d_position;
    Data<SeqElement<sofa::geometry::QuadraticEdge>> d_quadraticEdges;
    Data<SeqElement<sofa::geometry::QuadraticTriangle>> d_quadraticTriangles;
    Data<SeqElement<sofa::geometry::QuadraticQuad>> d_quadraticQuads;
    Data<SeqElement<sofa::geometry::QuadraticTetrahedron>> d_quadraticTetrahedra;
    Data<SeqElement<sofa::geometry::QuadraticHexahedron>> d_quadraticHexahedra;

protected:
    void doUpdate() override;

    LinearToHigherOrderElements();
};

}  // namespace sofa::component::engine::generate
