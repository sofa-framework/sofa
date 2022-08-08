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
#include <sofa/core/topology/TopologyData.h>
#include <sofa/type/RGBAColor.h>

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

class SOFA_GL_COMPONENT_RENDERING3D_API PointSplatModel : public core::visual::VisualModel
{
public:
    SOFA_CLASS(PointSplatModel,core::visual::VisualModel);
protected:
    PointSplatModel();
    ~PointSplatModel() override;
public:
    void init() override;

    void reinit() override;

    virtual bool isTransparent() {return true;}

    void drawTransparent(const core::visual::VisualParams* vparams) override;

private:
    Data<float>		radius; ///< Radius of the spheres.
    Data<int>		textureSize; ///< Size of the billboard texture.
    Data<float>		alpha; ///< Opacity of the billboards. 1.0 is 100% opaque.
    Data<sofa::type::RGBAColor>	color; ///< Billboard color.(default=[1.0,1.0,1.0,1.0])

    core::topology::BaseMeshTopology*	_topology;
    core::behavior::BaseMechanicalState* _mstate;

    unsigned char *texture_data;
    sofa::core::topology::PointData<sofa::type::vector<unsigned char> >		pointData; ///< scalar field modulating point colors

    typedef defaulttype::Vec3fTypes::Coord Coord;
    typedef defaulttype::Vec3fTypes::VecCoord VecCoord;
    typedef defaulttype::Vec3fTypes::Real Real;
};

} // namespace sofa::gl::component::rendering3d
