/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_ENGINE_MeshClosingEngine_H
#define SOFA_COMPONENT_ENGINE_MeshClosingEngine_H

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/component.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa
{

namespace component
{

namespace engine
{

/**
 * This class closes a triangle mesh and provides :
 * - a closed mesh (outputPosition, and outputTriangles)
 * - a index list of closing points (in closed mesh)
 * - a mesh of the closing (outputPosition, and outputTriangles)
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
    typedef typename core::topology::BaseMeshTopology::SetIndex SeqIndices;

    /// inputs
    Data< SeqPositions > position;
    Data< SeqTriangles > triangles;
    Data< SeqQuads > quads;

    /// outputs
    Data< SeqPositions > outputPosition;
    Data< SeqTriangles > outputTriangles;
    Data< SeqQuads > outputQuads;
    Data< SeqIndices > indices;
    Data< SeqPositions > closingPosition;
    Data< SeqTriangles > closingTriangles;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const MeshClosingEngine<DataTypes>* = NULL) { return DataTypes::Name();    }

protected:

    MeshClosingEngine()    : Inherited()
      , position(initData(&position,"position","input vertices"))
      , triangles(initData(&triangles,"triangles","input triangles"))
      , quads(initData(&quads,"quads","input quads"))
      , outputPosition(initData(&outputPosition,"outputPosition","Vertices of closed mesh"))
      , outputTriangles(initData(&outputTriangles,"outputTriangles","Triangles of closed mesh"))
      , outputQuads(initData(&outputQuads,"outputQuads","Quads of closed mesh (=input quads with current method)"))
      , indices(initData(&indices,"indices","output indices of closing vertices "))
      , closingPosition(initData(&closingPosition,"closingPosition","Vertices of the closing parts"))
      , closingTriangles(initData(&closingTriangles,"closingTriangles","Triangles of the closing parts"))
    {
    }

    virtual ~MeshClosingEngine() {}

public:
    virtual void init()
    {
        addInput(&position);
        addInput(&triangles);
        addInput(&quads);
        addOutput(&outputPosition);
        addOutput(&outputTriangles);
        addOutput(&outputQuads);
        addOutput(&indices);
        addOutput(&closingPosition);
        addOutput(&closingTriangles);
        setDirtyValue();
    }

    virtual void reinit()    { update();  }
    void update();
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_MeshClosingEngine_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_ENGINE_API MeshClosingEngine<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_ENGINE_API MeshClosingEngine<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
