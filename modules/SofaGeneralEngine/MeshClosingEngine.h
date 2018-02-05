/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_ENGINE_MeshClosingEngine_H
#define SOFA_COMPONENT_ENGINE_MeshClosingEngine_H
#include "config.h"

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/SVector.h>

namespace sofa
{

namespace component
{

namespace engine
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
    typedef helper::SVector<typename core::topology::BaseMeshTopology::PointID> SeqIndex;
    typedef helper::vector<SeqIndex> VecSeqIndex;  ///< vector of index lists

    /// inputs
    Data< SeqPositions > inputPosition;
    Data< SeqTriangles > inputTriangles;
    Data< SeqQuads > inputQuads;

    /// outputs
    Data< SeqPositions > position;
    Data< SeqTriangles > triangles;
    Data< SeqQuads > quads;
    Data< VecSeqIndex > indices;
    Data< SeqPositions > closingPosition;
    Data< SeqTriangles > closingTriangles;

    virtual std::string getTemplateName() const    override { return templateName(this);    }
    static std::string templateName(const MeshClosingEngine<DataTypes>* = NULL) { return DataTypes::Name();    }

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
    }

    virtual ~MeshClosingEngine() {}

public:
    virtual void init() override
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
        setDirtyValue();
    }

    virtual void reinit()    override { update();  }
    void update() override;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_MeshClosingEngine_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API MeshClosingEngine<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API MeshClosingEngine<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
