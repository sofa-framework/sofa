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

#include <sofa/gl/component/rendering3d/config.h>

#include <sofa/core/visual/VisualModel.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/core/topology/TopologyData.h>
#include <sofa/gl/template.h>

namespace sofa::core::topology
{
    class BaseMeshTopology;
} // namespace sofa::core::topology

namespace sofa::core::behavior
{
    class BaseMechanicalState;
} // namespace sofa::core::behavior

namespace sofa::gl::component::rendering3d
{

class SOFA_GL_COMPONENT_RENDERING3D_API SlicedVolumetricModel : public core::visual::VisualModel
{
public:
    SOFA_CLASS(SlicedVolumetricModel, core::visual::VisualModel);
protected:
    SlicedVolumetricModel();
    ~SlicedVolumetricModel() override;
public:
    void init() override;

    void reinit() override;

    virtual bool isTransparent() {return true;}

    void drawTransparent(const core::visual::VisualParams* vparams) override;

protected:
//    void setColor(float r, float g, float b);
//    void setColor(std::string color);

    void findAndDrawTriangles();

    Data<float>		alpha; ///< Opacity of the billboards. 1.0 is 100% opaque.
    Data<sofa::type::RGBAColor>	color; ///< Billboard color.(default=1.0,1.0,1.0,1.0)

    Data<int> _nbPlanes; ///< Number of billboards.
    int _nbPlanesOld;

    core::topology::BaseMeshTopology*	_topology;
    core::behavior::BaseMechanicalState* _mstate;

    unsigned char *texture_data;

    typedef defaulttype::Vec3fTypes::Coord Coord;
    typedef defaulttype::Vec3fTypes::VecCoord VecCoord;
    typedef defaulttype::Vec3fTypes::Real Real;


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

} // namespace sofa::gl::component::rendering3d
