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
#include <sofa/type/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/SVector.h>

namespace sofa::component::engine::generate
{

/**
 * This class closes a triangle mesh and provides :
 * - a closed mesh (position, and triangles)
 * - a index list of closing points (in closed mesh)
 * - a mesh of the closing (position, and triangles)
 * @author benjamin gilles
 */
template <class DataTypes>
class MeshClosingEngine : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(MeshClosingEngine,DataTypes),Inherited);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef VecCoord SeqPositions;
    typedef typename core::topology::BaseMeshTopology::Triangle Triangle;
    typedef typename core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef typename core::topology::BaseMeshTopology::SeqQuads SeqQuads;
    typedef typename core::topology::BaseMeshTopology::PointID PointID;
    typedef type::SVector<typename core::topology::BaseMeshTopology::PointID> SeqIndex;
    typedef type::vector<SeqIndex> VecSeqIndex;  ///< vector of index lists

    /// inputs
    Data< SeqPositions > inputPosition;
    Data< SeqTriangles > inputTriangles; ///< input triangles
    Data< SeqQuads > inputQuads; ///< input quads

    /// outputs
    Data< SeqPositions > position;
    Data< SeqTriangles > triangles; ///< Triangles of closed mesh
    Data< SeqQuads > quads; ///< Quads of closed mesh (=input quads with current method)
    Data< VecSeqIndex > indices; ///< Index lists of the closing parts
    Data< SeqPositions > closingPosition; ///< Vertices of the closing parts
    Data< SeqTriangles > closingTriangles; ///< Triangles of the closing parts

protected:

    MeshClosingEngine()    : Inherited()
      , inputPosition(initData(&inputPosition,"inputPosition","input vertices"))
      , inputTriangles(initData(&inputTriangles,"inputTriangles","input triangles"))
      , inputQuads(initData(&inputQuads,"inputQuads","input quads"))
      , position(initData(&position,"position","Vertices of closed mesh"))
      , triangles(initData(&triangles,"triangles","Triangles of closed mesh"))
      , quads(initData(&quads,"quads","Quads of closed mesh (=input quads with current method)"))
      , indices(initData(&indices,"indices","Index lists of the closing parts"))
      , closingPosition(initData(&closingPosition,"closingPosition","Vertices of the closing parts"))
      , closingTriangles(initData(&closingTriangles,"closingTriangles","Triangles of the closing parts"))
    {
        addInput(&inputPosition);
        addInput(&inputTriangles);
        addInput(&inputQuads);
        addOutput(&position);
        addOutput(&triangles);
        addOutput(&quads);
        addOutput(&indices);
        addOutput(&closingPosition);
        addOutput(&closingTriangles);
    }

    ~MeshClosingEngine() override {}

public:
    void init() override
    {
        setDirtyValue();
    }

    void reinit()    override { update();  }
    void doUpdate() override;
};

#if !defined(SOFA_COMPONENT_ENGINE_MeshClosingEngine_CPP)
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API MeshClosingEngine<defaulttype::Vec3Types>;
 
#endif

} //namespace sofa::component::engine::generate
