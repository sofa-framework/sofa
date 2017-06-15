/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef PLUGINS_PIM_SCULPTBODYPERFORMER_CPP
#define PLUGINS_PIM_SCULPTBODYPERFORMER_CPP

#include "SculptBodyPerformer.inl"
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/Factory.inl>

namespace plugins
{

namespace pim
{

#ifndef SOFA_DOUBLE
//      template class SOFA_COMPONENT_MISC_API  SculptBodyPerformer<defaulttype::Vec3fTypes>;
#endif
#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_MISC_API  SculptBodyPerformer<defaulttype::Vec3dTypes>;
#endif

#ifndef WIN32
#ifndef SOFA_DOUBLE
//      helper::Creator<InteractionPerformer::InteractionPerformerFactory, SculptBodyPerformer<defaulttype::Vec3fTypes> >  SculptBodyPerformerVec3fClass("SculptBody",true);
#endif
#ifndef SOFA_FLOAT
helper::Creator<InteractionPerformer::InteractionPerformerFactory, SculptBodyPerformer<defaulttype::Vec3dTypes> >  SculptBodyPerformerVec3dClass("SculptBody",true);
#endif
#endif

}
}
#endif
