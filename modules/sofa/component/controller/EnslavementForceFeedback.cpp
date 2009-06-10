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

    mState = dynamic_cast<core::componentmodel::behavior::MechanicalState<Rigid3dTypes> *> (context->getMechanicalState());
    if (!mState)
        serr << "EnslavementForceFeedback has no binding MechanicalState" << sendl;

    // fetch all collision models in the scene starting from this node
    simulation::Node* context = dynamic_cast<simulation::Node*>(getContext());
    if(context)
        context->getTreeObjects<core::CollisionModel>(&collisionModels);
}

void EnslavementForceFeedback::computeForce(double x, double y, double z,
        double /*u*/, double /*v*/, double /*w*/, double /*q*/,
        double& fx, double& fy, double& fz)
{
    if (f_activate.getValue())
    {
        for (unsigned int i=0; i<collisionModels.size(); i++)
        {
            // find first contact and generate a force proportional to the difference of Phantom position and
            // contact position on object surface
            if (collisionModels[i]->getNumberOfContacts() > 0)
            {
                double mx = (*mState->getX())[0].getCenter()[0];
                double my = (*mState->getX())[0].getCenter()[1];
                double mz = (*mState->getX())[0].getCenter()[2];

                const double& s = stiffness.getValue();
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

int enslavementForceFeedbackClass = sofa::core::RegisterObject("Eslavement force feedback for the omni")
        .add< EnslavementForceFeedback >();

SOFA_DECL_CLASS(EnslavementForceFeedback)

} // namespace controller
} // namespace component
} // namespace sofa
