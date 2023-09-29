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
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::engine::generate
{

/**
 * This class generates a cylinder mesh given its radius, length and height. The output  mesh is composed of tetrahedra elements
 */
template <class DataTypes>
class GenerateGrid : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(GenerateGrid,DataTypes),core::DataEngine);
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef sofa::type::Vec<3,Real> Vec3;
    typedef sofa::type::Vec<3,size_t> Vec3Int;
    typedef sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
    typedef sofa::core::topology::BaseMeshTopology::SeqHexahedra SeqHexahedra;
    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef sofa::core::topology::BaseMeshTopology::SeqQuads SeqQuads;
    typedef typename SeqTetrahedra::value_type Tetrahedron;
    typedef typename SeqHexahedra::value_type Hexahedron;
    typedef typename SeqTriangles::value_type Triangle;
    typedef typename SeqQuads::value_type Quad;
    typedef sofa::core::topology::Topology::PointID PointID;

public:

    GenerateGrid();

    ~GenerateGrid() override {}

    void init() override;

    void reinit() override;

    void doUpdate() override;


public:
    Data<VecCoord> d_outputX; ///< ouput position
    Data<SeqTetrahedra> d_tetrahedron; ///< output tetrahedra
    Data<SeqQuads> d_quad; ///< output quads
    Data<SeqTriangles> d_triangle; ///< output triangles
    Data<SeqHexahedra> d_hexahedron; ///< output hexahedra
    Data<Vec3> d_minCorner; ///< the position of the minimum corner 
    Data<Vec3> d_maxCorner; ///< the position of the maximum corner 
    Data<Vec3Int> d_resolution; ///< the resolution in the 3 directions
};


#if !defined(SOFA_COMPONENT_ENGINE_GENERATEGRID_CPP)
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API GenerateGrid<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API GenerateGrid<defaulttype::Vec2Types>;

#endif

} //namespace sofa::component::engine::generate
