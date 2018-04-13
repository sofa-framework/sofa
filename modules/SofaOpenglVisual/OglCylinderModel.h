/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_VISUALMODEL_POINTSPLATMODEL_H
#define SOFA_COMPONENT_VISUALMODEL_POINTSPLATMODEL_H
#include "config.h"

#include <sofa/core/visual/VisualModel.h>
#include <SofaBaseVisual/VisualModelImpl.h>
#include <sofa/defaulttype/VecTypes.h>
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/defaulttype/RGBAColor.h>

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


// I have no idea what is Ogl in this component ?...
class SOFA_OPENGL_VISUAL_API OglCylinderModel : public core::visual::VisualModel, public ExtVec3fState
{
public:
    SOFA_CLASS2(OglCylinderModel,core::visual::VisualModel,ExtVec3fState);
protected:
    OglCylinderModel();
    virtual ~OglCylinderModel();
public:
    virtual void init() override;

    virtual void reinit() override;

    virtual void drawVisual(const core::visual::VisualParams* vparams) override;

    virtual void exportOBJ(std::string /*name*/, std::ostream* /*out*/, std::ostream* /*mtl*/, int& /*vindex*/, int& /*nindex*/, int& /*tindex*/, int& /*count*/) override;

private:
    void setColor(float r, float g, float b, float a);
    void setColor(std::string color);

private:
    Data<float>		radius; ///< Radius of the cylinder.
    // Data<float>		alpha;
    Data<defaulttype::RGBAColor>	color; ///< Color of the cylinders.

    typedef sofa::helper::vector<core::topology::Edge>  SeqEdges;
    Data<SeqEdges> d_edges; ///< List of edge indices


    float r,g,b,a;
    // component::topology::PointData<sofa::helper::vector<unsigned char> >		pointData;

    typedef defaulttype::ExtVec3fTypes::Coord Coord;
    typedef defaulttype::ExtVec3fTypes::VecCoord VecCoord;
    typedef defaulttype::ExtVec3fTypes::Real Real;

public:
    virtual bool insertInNode( core::objectmodel::BaseNode* node ) override { Inherit1::insertInNode(node); Inherit2::insertInNode(node); return true; }
    virtual bool removeInNode( core::objectmodel::BaseNode* node ) override { Inherit1::removeInNode(node); Inherit2::removeInNode(node); return true; }
};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
