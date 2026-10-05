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

#include <sofa/core/objectmodel/BaseComponent.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/core/behavior/BaseTimeIntegrator.h>

namespace sofa::core::behavior
{

/**
 *  \brief Component responsible for main animation algorithms, managing how
 *  and when mechanical computation happens in one animation step.
 *
 *  This class can optionally replace the default computation scheme of computing
 *  collisions then doing an integration step.
 *
 *  Note that it is in a preliminary stage, hence its functionalities and API will
 *  certainly change soon.
 *
 */
class SOFA_CORE_API BaseAnimationLoop : public BaseTimeIntegrator
{

public:
    SOFA_ABSTRACT_CLASS(BaseAnimationLoop, BaseTimeIntegrator);

protected:
    BaseAnimationLoop();

    ~BaseAnimationLoop() override;

private:
    BaseAnimationLoop(const BaseAnimationLoop& n) = delete ;
    BaseAnimationLoop& operator=(const BaseAnimationLoop& n) = delete ;

public:
    void init() override;


    virtual void integrate(const core::ExecParams* params, SReal dt) override;

    /// Main computation method.
    ///
    /// Specify and execute all computations for computing a timestep, such
    /// as one or more collisions and integrations stages.
    virtual void step(const core::ExecParams* params, SReal dt) = 0;

};

} // namespace sofa::core::behavior
