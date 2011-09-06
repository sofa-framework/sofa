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
#include <sofa/component/controller/EnslavementForceFeedback.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/core/CollisionModel.h>

using namespace std;
using namespace sofa::defaulttype;

namespace sofa
{
namespace component
{
namespace controller
{

EnslavementForceFeedback::EnslavementForceFeedback()
    : ForceFeedback(),
      stiffness(initData(&stiffness, 1.0, "stiffness", "Penalty stiffness"))
{}

EnslavementForceFeedback::~EnslavementForceFeedback()
{}

void EnslavementForceFeedback::init()
{
    this->ForceFeedback::init();

    mState = dynamic_cast<core::behavior::MechanicalState<Rigid3dTypes> *> (context->getMechanicalState());
    if (!mState)
        serr << "EnslavementForceFeedback has no binding MechanicalState" << sendl;

    // fetch all collision models in the scene starting from this node
    simulation::Node* context = dynamic_cast<simulation::Node*>(getContext());
    if(context)
        context->getTreeObjects<core::CollisionModel>(&collisionModels);
}

void EnslavementForceFeedback::computeForce(SReal x, SReal y, SReal z,
        SReal /*u*/, SReal /*v*/, SReal /*w*/, SReal /*q*/,
        SReal& fx, SReal& fy, SReal& fz)
{
    if (f_activate.getValue())
    {
        for (unsigned int i=0; i<collisionModels.size(); i++)
        {
            // find first contact and generate a force proportional to the difference of Phantom position and
            // contact position on object surface
            if (collisionModels[i]->getNumberOfContacts() > 0)
            {
                SReal mx = (*mState->getX())[0].getCenter()[0];
                SReal my = (*mState->getX())[0].getCenter()[1];
                SReal mz = (*mState->getX())[0].getCenter()[2];

                const SReal& s = stiffness.getValue();
                fx = s * (mx - x);
                fy = s * (my - y);
                fz = s * (mz - z);
                return;
            }
        }
    }
    fx = 0.0;
    fy = 0.0;
    fz = 0.0;
}


void EnslavementForceFeedback::computeWrench(const SolidTypes<SReal>::Transform &world_H_tool,
        const SolidTypes<SReal>::SpatialVector &/*V_tool_world*/,
        SolidTypes<SReal>::SpatialVector &W_tool_world )
{

    if (!f_activate.getValue())
    {
        W_tool_world.clear();
        return;
    }
    bool is_in_contact = false;
    for (unsigned int i=0; i<collisionModels.size(); i++)
    {
        // find first contact and generate a force proportional to the difference of interface position and
        // contact position on object surface
        if (collisionModels[i]->getNumberOfContacts() > 0)
        {
            is_in_contact = true;
        }
    }

    if (is_in_contact)
    {
        SolidTypes<SReal>::Transform world_H_tool_simu(  (*mState->getX())[0].getCenter(), (*mState->getX())[0].getOrientation()  );
        SolidTypes<SReal>::Transform tool_H_tool_simu = world_H_tool.inversed() * world_H_tool_simu;
        SolidTypes<SReal>::SpatialVector DX_tool_tool = tool_H_tool_simu.DTrans();

        SolidTypes<SReal>::SpatialVector W_tool_tool(DX_tool_tool.getLinearVelocity() * stiffness.getValue(),
                DX_tool_tool.getAngularVelocity() * angular_stiffness.getValue());

        W_tool_world.setForce (world_H_tool.projectVector( W_tool_tool.getForce() ) );
        W_tool_world.setTorque(world_H_tool.projectVector( W_tool_tool.getTorque() ) );
    }
    else
    {
        W_tool_world.clear();
    }




}

int enslavementForceFeedbackClass = sofa::core::RegisterObject("Eslavement force feedback for the haptic device")
        .add< EnslavementForceFeedback >();

SOFA_DECL_CLASS(EnslavementForceFeedback)

} // namespace controller
} // namespace component
} // namespace sofa
