/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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

using namespace std;

namespace sofa
{
namespace component
{
namespace controller
{

void EnslavementForceFeedback::init()
{
    this->ForceFeedback::init();
    OmniDriver* driver = context->get<OmniDriver>();
    cout << "init EnslavementForceFeedback" << driver << " done " << std::endl;

    driver->setForceFeedback(this);

    mState = dynamic_cast<MechanicalState<Rigid3dTypes> *> (this->getContext()->getMechanicalState());
    if (!mState)
        logWarning("EnslavementForceFeedback has no binding MechanicalState");
    simulation::tree::GNode* context = dynamic_cast<simulation::tree::GNode*>(getContext());
    context->getTreeObjects<core::CollisionModel>(&collisionModels);
}

void EnslavementForceFeedback::computeForce(double x, double y, double z, double u, double v, double w, double q, double& fx, double& fy, double& fz)
{
    if (f_activate.getValue())
    {
        for (unsigned int i=0; i<collisionModels.size(); i++)
        {
            if (collisionModels[i]->getNumberOfContacts() > 0)
            {
                double mx = (*mState->getX())[0].getCenter()[0];
                double my = (*mState->getX())[0].getCenter()[1];
                double mz = (*mState->getX())[0].getCenter()[2];

                fx = 3.0 * (mx - 	x);
                fy = 3.0 * (my - y);
                fz = 3.0 * (mz - z);
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
