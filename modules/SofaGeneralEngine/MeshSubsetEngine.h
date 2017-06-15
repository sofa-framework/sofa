/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_MeshSubsetEngine_H
#define SOFA_COMPONENT_ENGINE_MeshSubsetEngine_H
#include "config.h"

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa
{

namespace component
{

namespace engine
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
    Data< SeqEdges > inputEdges;
    Data< SeqTriangles > inputTriangles;
    Data< SeqQuads > inputQuads;
    Data< SetIndices > indices;

    /// outputs
    Data< SeqPositions > position;
    Data< SeqEdges > edges;
    Data< SeqTriangles > triangles;
    Data< SeqQuads > quads;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const MeshSubsetEngine<DataTypes>* = NULL) { return DataTypes::Name();    }

protected:

    MeshSubsetEngine()    : Inherited()
      , inputPosition(initData(&inputPosition,"inputPosition","input vertices"))
      , inputEdges(initData(&inputEdges,"inputEdges","input edges"))
      , inputTriangles(initData(&inputTriangles,"inputTriangles","input triangles"))
      , inputQuads(initData(&inputQuads,"inputQuads","input quads"))
      , indices(initData(&indices,"indices","Index lists of the selected vertices"))
      , position(initData(&position,"position","Vertices of mesh subset"))
      , edges(initData(&edges,"edges","edges of mesh subset"))
      , triangles(initData(&triangles,"triangles","Triangles of mesh subset"))
      , quads(initData(&quads,"quads","Quads of mesh subset"))
    {
    }

    virtual ~MeshSubsetEngine() {}

public:
    virtual void init()
    {
        addInput(&inputPosition);
        addInput(&inputEdges);
        addInput(&inputTriangles);
        addInput(&inputQuads);
        addInput(&indices);
        addOutput(&position);
        addOutput(&edges);
        addOutput(&triangles);
        addOutput(&quads);
        setDirtyValue();
    }

    virtual void reinit()    { update();  }
    void update();
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_MeshSubsetEngine_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API MeshSubsetEngine<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API MeshSubsetEngine<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
