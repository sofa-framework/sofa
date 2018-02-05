/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_SIMULATION_DEFAULTVISUALMANAGERLOOP_H
#define SOFA_SIMULATION_DEFAULTVISUALMANAGERLOOP_H

#include <sofa/core/visual/VisualLoop.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/simulation/simulationcore.h>
#include <sofa/simulation/Node.h>

namespace sofa
{

namespace core
{ 
namespace visual
{
class VisualParams;
} // namespace visual
} // namespace core

namespace simulation
{

/**
 *  \brief Default VisualManager Loop to be created when no VisualManager found on simulation::node.
 *
 *
 */

class SOFA_SIMULATION_CORE_API DefaultVisualManagerLoop : public core::visual::VisualLoop
{
public:
    typedef core::visual::VisualLoop Inherit;
    typedef sofa::core::objectmodel::BaseContext BaseContext;
    typedef sofa::core::objectmodel::BaseObjectDescription BaseObjectDescription;
    SOFA_CLASS(DefaultVisualManagerLoop,core::visual::VisualLoop);
protected:
    DefaultVisualManagerLoop(simulation::Node* gnode = NULL);

    virtual ~DefaultVisualManagerLoop();
public:
    virtual void init() override;

    /// Initialize the textures
    virtual void initStep(sofa::core::ExecParams* params) override;

    /// Update the Visual Models: triggers the Mappings
    virtual void updateStep(sofa::core::ExecParams* params) override;

    /// Update contexts. Required before drawing the scene if root flags are modified.
    virtual void updateContextStep(sofa::core::visual::VisualParams* vparams) override;

    /// Render the scene
    virtual void drawStep(sofa::core::visual::VisualParams* vparams) override;

    /// Compute the bounding box of the scene. If init is set to "true", then minBBox and maxBBox will be initialised to a default value
    virtual void computeBBoxStep(sofa::core::visual::VisualParams* vparams, SReal* minBBox, SReal* maxBBox, bool init) override;


    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T*, BaseContext* context, BaseObjectDescription* arg)
    {
        simulation::Node* gnode = dynamic_cast<simulation::Node*>(context);
        typename T::SPtr obj = sofa::core::objectmodel::New<T>(gnode);
        if (context) context->addObject(obj);
        if (arg) obj->parse(arg);
        return obj;
    }

protected:

    simulation::Node* gRoot;
};

} // namespace simulation

} // namespace sofa

#endif  /* SOFA_SIMULATION_DEFAULTVISUALMANAGERLOOP_H */
