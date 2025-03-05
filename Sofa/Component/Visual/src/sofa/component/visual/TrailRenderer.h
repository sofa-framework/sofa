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
#include <queue>

namespace sofa::component::visual
{

/**
 * Render a trail behind particles
 *
 * It can be used to draw the trajectory of a dof.
 * This component does not support topological changes (point removal or point addition) and
 * list reordering.
 */
template<class DataTypes>
class TrailRenderer : public core::visual::VisualModel
{
public:
    SOFA_CLASS(TrailRenderer, core::visual::VisualModel);

    using Coord = typename DataTypes::Coord;

    Data< sofa::type::vector<Coord> > d_position; ///< Position of the particles behind which a trail is rendered
    Data< sofa::Size > d_nbSteps; ///< Number of time steps to use to render the trail
    Data<sofa::type::RGBAColor> d_color; ///< Color of the trail
    Data<float> d_thickness; ///< Thickness of the trail


    void handleEvent(core::objectmodel::Event *) override;
    void doDrawVisual(const core::visual::VisualParams* vparams) override;
    void reset() override;

protected:

    TrailRenderer();

    void storeParticlePositions();
    void removeFirstElements();

    type::vector<std::vector<sofa::type::Vec3> > m_trail;
};

#if !defined(SOFA_COMPONENT_VISUAL_TRAILRENDERER_CPP)
extern template class SOFA_COMPONENT_VISUAL_API TrailRenderer<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_VISUAL_API TrailRenderer<defaulttype::Rigid3Types>;
#endif

}
