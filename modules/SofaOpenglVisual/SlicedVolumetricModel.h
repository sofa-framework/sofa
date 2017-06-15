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
#ifndef SOFA_COMPONENT_VISUALMODEL_SlicedVolumetricModel_H
#define SOFA_COMPONENT_VISUALMODEL_SlicedVolumetricModel_H
#include "config.h"

#include <sofa/core/visual/VisualModel.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RGBAColor.h>
#include <SofaBaseTopology/TopologyData.h>

#include <sofa/helper/gl/template.h>

namespace sofa
{
namespace core
{
namespace topology
{
class BaseMeshTopology;
}
namespace behavior
{
class BaseMechanicalState;
}
}

namespace component
{

namespace visualmodel
{

class SOFA_OPENGL_VISUAL_API SlicedVolumetricModel : public core::visual::VisualModel
{
public:
    SOFA_CLASS(SlicedVolumetricModel, core::visual::VisualModel);
protected:
    SlicedVolumetricModel();
    virtual ~SlicedVolumetricModel();
public:
    virtual void init();

    virtual void reinit();

    virtual bool isTransparent() {return true;}

    virtual void drawTransparent(const core::visual::VisualParams* vparams);

protected:
//    void setColor(float r, float g, float b);
//    void setColor(std::string color);

    void findAndDrawTriangles();

    Data<float>		alpha;
    Data<defaulttype::RGBAColor>	color;

    Data<int> _nbPlanes;
    int _nbPlanesOld;

    core::topology::BaseMeshTopology*	_topology;
    core::behavior::BaseMechanicalState* _mstate;

    unsigned char *texture_data;

    typedef defaulttype::ExtVec3fTypes::Coord Coord;
    typedef defaulttype::ExtVec3fTypes::VecCoord VecCoord;
    typedef defaulttype::ExtVec3fTypes::Real Real;


    bool _first;
    GLuint _texname;
    int _width,_height,_depth;
    Coord vRight,vUp,_planeNormal;
    Real _radius;
    Real _planeSeparations;
    void computePlaneSeparations();

    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef std::pair< Coord , Coord > Intersection; // position, texture coord
    typedef std::map< Edge, Intersection > EdgesMap;

    static const int __edges__[12][2];

    int intersectionSegmentPlane( const Coord&s0,const Coord&s1, const Coord&segmentDirection, const Coord& planeNormal, const Real& planeConstant, Real & m_fLineT );

    VecCoord _textureCoordinates;

    double _minBBox[3], _maxBBox[3];
};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
