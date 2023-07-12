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
#include <sofa/core/visual/VisualState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/topology/TopologyData.h>
#include <sofa/core/topology/Topology.h>
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

// I have no idea what is Ogl in this component ?...
class SOFA_GL_COMPONENT_RENDERING3D_API OglCylinderModel : public core::visual::VisualModel, public sofa::core::visual::VisualState<defaulttype::Vec3Types>
{
public:
    using Vec3State = sofa::core::visual::VisualState<defaulttype::Vec3Types>;
    SOFA_CLASS2(OglCylinderModel,core::visual::VisualModel, Vec3State);

    using Index = sofa::Index;
protected:
    OglCylinderModel();
    ~OglCylinderModel() override;
public:
    void init() override;

    void reinit() override;

    void doDrawVisual(const core::visual::VisualParams* vparams) override;

    void exportOBJ(std::string /*name*/, std::ostream* /*out*/, std::ostream* /*mtl*/, Index& /*vindex*/, Index& /*nindex*/, Index& /*tindex*/, int& /*count*/) override;

private:
    void setColor(float r, float g, float b, float a);
    void setColor(std::string color);

private:
    Data<float>		radius; ///< Radius of the cylinder.
    // Data<float>		alpha;
    Data<sofa::type::RGBAColor>	color; ///< Color of the cylinders.

    typedef sofa::type::vector<core::topology::Edge>  SeqEdges;
    Data<SeqEdges> d_edges; ///< List of edge indices


    float r,g,b,a;
    // sofa::core::topology::PointData<sofa::type::vector<unsigned char> >		pointData;

    typedef Vec3State::Coord Coord;
    typedef Vec3State::VecCoord VecCoord;
    typedef Vec3State::Real Real;

public:
    bool insertInNode( core::objectmodel::BaseNode* node ) override { Inherit1::insertInNode(node); Inherit2::insertInNode(node); return true; }
    bool removeInNode( core::objectmodel::BaseNode* node ) override { Inherit1::removeInNode(node); Inherit2::removeInNode(node); return true; }
};

} // namespace sofa::gl::component::rendering3d
