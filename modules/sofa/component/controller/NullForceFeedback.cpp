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
#include <sofa/component/controller/NullForceFeedback.h>
#include <sofa/core/ObjectFactory.h>

using namespace std;

namespace sofa
{
namespace component
{
namespace controller
{

void NullForceFeedback::init()
{
    this->ForceFeedback::init();
    OmniDriver* driver = context->get<OmniDriver>();
    sout << "init NullForceFeedback" << driver << " done " << sendl;
};

void NullForceFeedback::computeForce(double /*x*/, double /*y*/, double /*z*/, double /*u*/, double /*v*/, double /*w*/, double /*q*/, double& fx, double& fy, double& fz)
{
    fx = fy = fz = 0.0;
};



int nullForceFeedbackClass = sofa::core::RegisterObject("Null force feedback for the omni")
        .add< NullForceFeedback >();

SOFA_DECL_CLASS(NullForceFeedback)

} // namespace controller
} // namespace component
} // namespace sofa
