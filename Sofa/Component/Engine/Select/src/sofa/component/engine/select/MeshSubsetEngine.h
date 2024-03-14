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
#include <sofa/component/engine/select/config.h>

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/type/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa::component::engine::select
{

/**
 * This class extracts a mesh subset based on selected vertices
 */
template <class DataTypes>
class MeshSubsetEngine : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(MeshSubsetEngine,DataTypes),Inherited);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef VecCoord SeqPositions;
    typedef typename core::topology::BaseMeshTopology::Edge Edge;
    typedef typename core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef typename core::topology::BaseMeshTopology::Triangle Triangle;
    typedef typename core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef typename core::topology::BaseMeshTopology::Quad Quad;
    typedef typename core::topology::BaseMeshTopology::SeqQuads SeqQuads;
    typedef typename core::topology::BaseMeshTopology::PointID PointID;
    typedef typename core::topology::BaseMeshTopology::SetIndices SetIndices;

    /// inputs
    Data< SeqPositions > inputPosition;
    Data< SeqEdges > inputEdges; ///< input edges
    Data< SeqTriangles > inputTriangles; ///< input triangles
    Data< SeqQuads > inputQuads; ///< input quads
    Data< SetIndices > indices; ///< Index lists of the selected vertices

    /// outputs
    Data< SeqPositions > position;
    Data< SeqEdges > edges; ///< edges of mesh subset
    Data< SeqTriangles > triangles; ///< Triangles of mesh subset
    Data< SeqQuads > quads; ///< Quads of mesh subset

protected:
    MeshSubsetEngine();
    ~MeshSubsetEngine() override;

public:
    void init() override
    {
        setDirtyValue();
    }

    void reinit()    override { update();  }
    void doUpdate() override;
};

#if !defined(SOFA_COMPONENT_ENGINE_MeshSubsetEngine_CPP)
extern template class SOFA_COMPONENT_ENGINE_SELECT_API MeshSubsetEngine<defaulttype::Vec3Types>;
 
#endif

} //namespace sofa::component::engine::select
