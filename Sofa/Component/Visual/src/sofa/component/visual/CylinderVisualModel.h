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
#include <sofa/component/visual/config.h>

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/VisualState.h>
#include <sofa/type/RGBAColor.h>

namespace sofa::component::visual
{

class SOFA_COMPONENT_VISUAL_API CylinderVisualModel :
    public core::visual::VisualModel, public sofa::core::visual::VisualState<defaulttype::Vec3Types>
{
public:
    using Vec3State = sofa::core::visual::VisualState<defaulttype::Vec3Types>;
    SOFA_CLASS2(CylinderVisualModel,core::visual::VisualModel, Vec3State);

protected:
    CylinderVisualModel();
    ~CylinderVisualModel() override;
public:
    void init() override;

    void doDrawVisual(const core::visual::VisualParams* vparams) override;

    void exportOBJ(std::string /*name*/, std::ostream* /*out*/, std::ostream* /*mtl*/, Index& /*vindex*/, Index& /*nindex*/, Index& /*tindex*/, int& /*count*/) override;

private:
    Data<float>		radius; ///< Radius of the cylinder.
    Data<sofa::type::RGBAColor>	color; ///< Color of the cylinders.

    typedef sofa::type::vector<core::topology::Edge>  SeqEdges;
    Data<SeqEdges> d_edges; ///< List of edge indices

public:
    bool insertInNode( core::objectmodel::BaseNode* node ) override { Inherit1::insertInNode(node); Inherit2::insertInNode(node); return true; }
    bool removeInNode( core::objectmodel::BaseNode* node ) override { Inherit1::removeInNode(node); Inherit2::removeInNode(node); return true; }
};

} // namespace sofa::component::visual
