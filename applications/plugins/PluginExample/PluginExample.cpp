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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "PluginExample.h"
#include <sofa/core/Plugin.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

#include "MyBehaviorModel.h"
#include "MyMappingPendulumInPlane.inl"
#include "MyProjectiveConstraintSet.inl"

using namespace sofa::defaulttype;
using sofa::component::behaviormodel::MyBehaviorModel;
using sofa::component::mapping::MyMappingPendulumInPlane;
using sofa::component::projectiveconstraintset::MyProjectiveConstraintSet;


class PluginExample: public sofa::core::Plugin {
public:
    PluginExample(): sofa::core::Plugin("PluginExample") {
        setDescription("A simple example of a SOFA plugin.");
        setVersion("0.1");
        setLicense("LGPL");
        setAuthors("The SOFA team");

        addComponent<MyBehaviorModel>("???");

        // Default instance of MyMappingPendulumInPlane
#ifdef SOFA_FLOAT
        addComponent< MyMappingPendulumInPlane<Vec1fTypes,Vec3fTypes> >();
#else
        addComponent< MyMappingPendulumInPlane<Vec1dTypes,Vec3dTypes> >();
#endif
        setDescription("MyMappingPendulumInPlane", "Mapping from an angle to a point in 2D.");

        // Other instances of MyMappingPendulumInPlane
#ifdef SOFA_FLOAT
        addTemplateInstance< MyMappingPendulumInPlane<Vec1fTypes,Vec2fTypes> >();
#elif !defined(SOFA_DOUBLE)
        addTemplateInstance< MyMappingPendulumInPlane<Vec1dTypes,Vec2dTypes> >();
        addTemplateInstance< MyMappingPendulumInPlane<Vec1fTypes,Vec3fTypes> >();
        addTemplateInstance< MyMappingPendulumInPlane<Vec1fTypes,Vec2fTypes> >();
#endif


        // Default instance of MyProjectiveConstraintSet
#ifdef SOFA_FLOAT
        addComponent< MyProjectiveConstraintSet<Vec3fTypes> >();
#else
        addComponent< MyProjectiveConstraintSet<Vec3dTypes> >();
#endif
        setDescription("MyProjectiveConstraintSet", "Just an example of templated component.");

        // Other instances of MyProjectiveConstraintSet
#ifdef SOFA_FLOAT
        addTemplateInstance< MyProjectiveConstraintSet<Vec1fTypes> >();
        addTemplateInstance< MyProjectiveConstraintSet<Rigid3fTypes> >();
#elif !defined(SOFA_DOUBLE)
        addTemplateInstance< MyProjectiveConstraintSet<Vec3dTypes> >();
        addTemplateInstance< MyProjectiveConstraintSet<Vec1dTypes> >();
        addTemplateInstance< MyProjectiveConstraintSet<Rigid3dTypes> >();
#endif
    }
};

SOFA_PLUGIN(PluginExample);
