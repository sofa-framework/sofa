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

#include <sofa/simulation/config.h>               // config.h *must*  be the first include
#include <sofa/simulation/fwd.h>
#include <sofa/core/visual/VisualLoop.h>
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa::core::objectmodel
{
// Forward declaration for extern template declaration. This design permit to
// not #include<sofa::simulation::Node>
extern template class SingleLink< sofa::simulation::DefaultVisualManagerLoop, simulation::Node, BaseLink::FLAG_STOREPATH>;
}

namespace sofa::simulation
{

/**
 *  \brief Default VisualManager Loop to be created when no VisualManager found on simulation::node.
 *
 */
class SOFA_SIMULATION_CORE_API DefaultVisualManagerLoop : public sofa::core::visual::VisualLoop
{
public:
    typedef sofa::core::visual::VisualLoop Inherit;
    typedef sofa::core::objectmodel::BaseContext BaseContext;
    typedef sofa::core::objectmodel::BaseObjectDescription BaseObjectDescription;
    SOFA_CLASS(DefaultVisualManagerLoop,sofa::core::visual::VisualLoop);
protected:
    DefaultVisualManagerLoop();
    ~DefaultVisualManagerLoop() override;
public:
    void init() override;

    /// Initialize the textures
    void initStep(sofa::core::ExecParams* params) override;

    /// Update the Visual Models: triggers the Mappings
    void updateStep(sofa::core::ExecParams* params) override;

    /// Update contexts. Required before drawing the scene if root flags are modified.
    void updateContextStep(sofa::core::visual::VisualParams* vparams) override;

    /// Render the scene
    void drawStep(sofa::core::visual::VisualParams* vparams) override;

    /// Compute the bounding box of the scene. If init is set to "true", then minBBox and maxBBox will be initialised to a default value
    void computeBBoxStep(sofa::core::visual::VisualParams* vparams, SReal* minBBox, SReal* maxBBox, bool init) override;

    SingleLink< DefaultVisualManagerLoop, simulation::Node, BaseLink::FLAG_STOREPATH> l_node;  ///< Link to the scene's node where the rendering will take place
};

} // namespace sofa
