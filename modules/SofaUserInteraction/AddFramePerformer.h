/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_COLLISION_ADDFRAMEPERFORMER_H
#define SOFA_COMPONENT_COLLISION_ADDFRAMEPERFORMER_H
#include "config.h"

#include <SofaUserInteraction/InteractionPerformer.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <../applications/plugins/frame/Blending.h>


namespace sofa
{

namespace component
{

namespace collision
{
using sofa::defaulttype::StdRigidTypes;

template <class DataTypes>
class AddFramePerformer: public TInteractionPerformer<DataTypes>
{
    typedef typename sofa::defaulttype::BaseFrameBlendingMapping<true> FBMapping;

public:
    AddFramePerformer(BaseMouseInteractor *i);
    ~AddFramePerformer();

    void start();
    void execute();

};



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_ADDFRAMEPERFORMER_CPP)
#ifndef SOFA_DOUBLE
extern template class SOFA_USER_INTERACTION_API  AddFramePerformer<defaulttype::Vec3fTypes>;
#endif
#ifndef SOFA_FLOAT
extern template class SOFA_USER_INTERACTION_API  AddFramePerformer<defaulttype::Vec3dTypes>;
#endif
#endif


}
}
}

#endif
