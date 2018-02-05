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
#ifndef SOFA_CORE_VISUAL_VISUALLOOP_H
#define SOFA_CORE_VISUAL_VISUALLOOP_H

#include <sofa/core/visual/VisualModel.h>

namespace sofa
{

namespace core
{

namespace visual
{

class VisualParams;

/*
 * VisualLoop is an API managing steps for drawing, rendering scene.
 * Components inherit from this API need to be unique in the root node of the scene
 * These components launch all visual visitor and managing visual steps.
 *
 * */
class SOFA_CORE_API VisualLoop : public VisualModel
{
public:
    SOFA_CLASS(VisualLoop, VisualModel);
    SOFA_BASE_CAST_IMPLEMENTATION(VisualLoop)
protected:
    /// Destructor
    virtual ~VisualLoop() { }
public:
    /// Initialize the textures
    virtual void initStep(sofa::core::ExecParams* /*params*/) {}

    /// Update the Visual Models: triggers the Mappings
    virtual void updateStep(sofa::core::ExecParams* /*params*/) {}

    /// Update contexts. Required before drawing the scene if root flags are modified.
    virtual void updateContextStep(sofa::core::visual::VisualParams* /*vparams*/) {}

    /// Render the scene
    virtual void drawStep(sofa::core::visual::VisualParams* /*vparams*/) {}

    /// Compute the bounding box of the scene. If init is set to "true", then minBBox and maxBBox will be initialised to a default value
    virtual void computeBBoxStep(sofa::core::visual::VisualParams* /*vparams*/, SReal* /*minBBox*/, SReal* /*maxBBox*/, bool /*init*/) { serr << "WARNING, VisualLoop::computeBBoxStep does nothing" << sendl; }

    virtual bool insertInNode( objectmodel::BaseNode* node ) override;
    virtual bool removeInNode( objectmodel::BaseNode* node ) override;
};

} // namespace visual

} // namespace core

} // namespace sofa

#endif /* SOFA_CORE_VISUAL_VISUALLOOP_H */
