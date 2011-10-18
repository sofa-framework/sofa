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
#include <sofa/component/controller/LCPForceFeedback.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/LCPcalc.h>
#include <sofa/defaulttype/RigidTypes.h>

using namespace std;
using namespace sofa::defaulttype;

namespace sofa
{
namespace component
{
namespace controller
{
int lCPForceFeedbackClass = sofa::core::RegisterObject("LCP force feedback for the device")
#ifndef SOFA_FLOAT
        .add< LCPForceFeedback<sofa::defaulttype::Vec1dTypes> >()
        .add< LCPForceFeedback<sofa::defaulttype::Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< LCPForceFeedback<sofa::defaulttype::Vec1fTypes> >()
        .add< LCPForceFeedback<sofa::defaulttype::Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_HAPTICS_API LCPForceFeedback<Vec1dTypes>;
template class SOFA_HAPTICS_API LCPForceFeedback<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_HAPTICS_API LCPForceFeedback<Vec1fTypes>;
template class SOFA_HAPTICS_API LCPForceFeedback<Rigid3fTypes>;
#endif
SOFA_DECL_CLASS(LCPForceFeedback)


} // namespace controller
} // namespace component
} // namespace sofa
