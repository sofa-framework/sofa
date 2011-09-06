/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_SIMULATION_DEFAULTVISUALMANAGERLOOP_H
#define SOFA_SIMULATION_DEFAULTVISUALMANAGERLOOP_H

#include <sofa/core/visual/VisualManager.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/common/common.h>
#include <sofa/simulation/common/Node.h>

using namespace sofa::core::objectmodel;
using namespace sofa::core::behavior;

namespace sofa
{

namespace simulation
{

/**
 *  \brief Default VisualManager Loop to be created when no VisualManager found on simulation::node.
 *
 *
 */

class SOFA_SIMULATION_COMMON_API DefaultVisualManagerLoop : public core::visual::VisualManager
{
public:
    typedef core::visual::VisualManager Inherit;
    SOFA_CLASS(DefaultVisualManagerLoop,core::visual::VisualManager);

    DefaultVisualManagerLoop(simulation::Node* gnode);

    virtual ~DefaultVisualManagerLoop();

    /// Initialize the textures
    virtual void initTextures(Node* root) {}

    /// Update the Visual Models: triggers the Mappings
    virtual void updateVisual(Node* root, double dt=0.0) {}

    /// Update contexts. Required before drawing the scene if root flags are modified.
    virtual void updateVisualContext(Node* root) {}

    /// Compute the bounding box of the scene. If init is set to "true", then minBBox and maxBBox will be initialised to a default value
    virtual void computeBBox(Node* root, SReal* minBBox, SReal* maxBBox, bool init=true) {}

    /// Render the scene
    virtual void draw(sofa::core::visual::VisualParams* vparams, Node* root) {}



    /// Construction method called by ObjectFactory.
    template<class T>
    static void create(T*& obj, BaseContext* context, BaseObjectDescription* arg)
    {
        simulation::Node* gnode = dynamic_cast<simulation::Node*>(context);
        obj = new T(gnode);
        if (context) context->addObject(obj);
        if (arg) obj->parse(arg);
    }

protected:

    simulation::Node* gnode;
};

} // namespace simulation

} // namespace sofa

#endif  /* SOFA_SIMULATION_DEFAULTVISUALMANAGERLOOP_H */
