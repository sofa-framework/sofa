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
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/visual/DrawMesh.h>
#include <sofa/core/visual/VisualModel.h>

namespace sofa::component::visual
{

class VisualMesh : public core::visual::VisualModel
{
public:
    SOFA_CLASS(VisualMesh, core::visual::VisualModel);

    Data<type::vector<type::Vec3>> d_position;
    Data<SReal> d_elementSpace;

    /// The topology will give access to the elements
    sofa::SingleLink<VisualMesh, sofa::core::topology::BaseMeshTopology,
        sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK> l_topology;

    void init() override;

protected:

    VisualMesh();

    void doDrawVisual(const core::visual::VisualParams* vparams) override;

    core::visual::DrawMesh m_drawMesh;

    void validateTopology();
};

}
