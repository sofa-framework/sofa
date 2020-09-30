/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "config.h"

#include <sofa/core/visual/VisualModel.h>
#include <sofa/defaulttype/VecTypes.h>
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/defaulttype/RGBAColor.h>

namespace sofa::component::visualmodel
{

using sofa::core::visual::VisualModel;
using sofa::defaulttype::RGBAColor;
using sofa::defaulttype::Vec3;
using sofa::helper::vector;

class SOFA_OPENGL_VISUAL_API PointSplatRenderer : public VisualModel
{
public:
    SOFA_CLASS(PointSplatRenderer,VisualModel);

    void init() override;
    void reinit() override;

    void drawTransparent(const core::visual::VisualParams* vparams) override;

    Data<float>         d_radius; ///< Radius of the spheres.
    Data<size_t>		d_textureSize; ///< Size of the billboard texture.

    Data<vector<Vec3>> d_points;
    Data<vector<Vec3>> d_normals;
    Data<vector<RGBAColor>> d_colors;

protected:
    PointSplatRenderer();
    ~PointSplatRenderer() override;

private:
    unsigned char *texture_data;
};

} /// sofa::component::visualmodel
