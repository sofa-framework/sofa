/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_ANIMATIONLOOP_MULTISTEPANIMATIONLOOP_H
#define SOFA_COMPONENT_ANIMATIONLOOP_MULTISTEPANIMATIONLOOP_H

#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/simulation/common/CollisionAnimationLoop.h>
#include <sofa/SofaBase.h>

namespace sofa
{

namespace component
{

namespace animationloop
{

class SOFA_BASE_ANIMATION_LOOP_API MultiStepAnimationLoop : public sofa::simulation::CollisionAnimationLoop
{
public:
    typedef sofa::simulation::CollisionAnimationLoop Inherit;
    SOFA_CLASS(MultiStepAnimationLoop, sofa::simulation::CollisionAnimationLoop);
protected:
    MultiStepAnimationLoop(simulation::Node* gnode);

    virtual ~MultiStepAnimationLoop();
public:
    virtual void step (const sofa::core::ExecParams* params /* PARAMS FIRST */, double dt);

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

    Data<int> collisionSteps;
    Data<int> integrationSteps;
};

} // namespace animationloop

} // namespace component

} // namespace sofa

#endif /* SOFA_COMPONENT_ANIMATIONLOOP_MULTISTEPANIMATIONLOOP_H */
