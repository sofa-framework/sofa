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
#include "Binding_DisplayFlagsData.h"
#include "Binding_Data.h"

#include <sofa/core/visual/DisplayFlags.h>
using namespace sofa::core::visual;
using namespace sofa::core::objectmodel;

SP_CLASS_METHODS_BEGIN(DisplayFlagsData)
SP_CLASS_METHODS_END

/*
\item showAll
\item showVisual
\item showVisualModels
\item showBehavior
\item showBehaviorModels
\item showForceFields
\item showInteractionForceFields
\item showCollision
\item showCollisionModels
\item showBoundingCollisionModels
\item showMapping
\item showMappings
\item showMechanicalMappings
\item showOptions
\item showWireFrame
\item showNormals
*/


#define DISPLAYFLAG_ATTRIBUTE_IMPL(flagName) \
    extern "C" PyObject * DisplayFlagsData_getAttr_show##flagName(PyObject *self, void*) \
    { \
        Data<DisplayFlags>* data= down_cast<Data<DisplayFlags> >( ((PyPtr<BaseData>*)self)->object ); \
        const DisplayFlags& flags = data->getValue(); \
        bool b = (tristate::false_value != flags.getShow##flagName()); \
        return PyBool_FromLong(b); \
    } \
    extern "C" int DisplayFlagsData_setAttr_show##flagName(PyObject *self, PyObject * args, void*) \
    { \
        Data<DisplayFlags>* data= down_cast<Data<DisplayFlags> >( ((PyPtr<BaseData>*)self)->object ); \
        bool b = (Py_True==args); \
        DisplayFlags* flags = data->beginEdit(); \
        flags->setShow##flagName(b); \
        /*printf("%s %d\n",#flagName,b);*/ \
        data->endEdit(); \
        return 0; \
    }

DISPLAYFLAG_ATTRIBUTE_IMPL(All)
DISPLAYFLAG_ATTRIBUTE_IMPL(Visual)
DISPLAYFLAG_ATTRIBUTE_IMPL(VisualModels)
DISPLAYFLAG_ATTRIBUTE_IMPL(Behavior)
DISPLAYFLAG_ATTRIBUTE_IMPL(BehaviorModels)
DISPLAYFLAG_ATTRIBUTE_IMPL(ForceFields)
DISPLAYFLAG_ATTRIBUTE_IMPL(InteractionForceFields)
DISPLAYFLAG_ATTRIBUTE_IMPL(Collision)
DISPLAYFLAG_ATTRIBUTE_IMPL(CollisionModels)
DISPLAYFLAG_ATTRIBUTE_IMPL(BoundingCollisionModels)
DISPLAYFLAG_ATTRIBUTE_IMPL(Mapping)
DISPLAYFLAG_ATTRIBUTE_IMPL(Mappings)
DISPLAYFLAG_ATTRIBUTE_IMPL(MechanicalMappings)
DISPLAYFLAG_ATTRIBUTE_IMPL(Options)
DISPLAYFLAG_ATTRIBUTE_IMPL(WireFrame)
DISPLAYFLAG_ATTRIBUTE_IMPL(Normals)



SP_CLASS_ATTRS_BEGIN(DisplayFlagsData)
SP_CLASS_ATTR(DisplayFlagsData,showAll)
SP_CLASS_ATTR(DisplayFlagsData,showVisual)
SP_CLASS_ATTR(DisplayFlagsData,showVisualModels)
SP_CLASS_ATTR(DisplayFlagsData,showBehavior)
SP_CLASS_ATTR(DisplayFlagsData,showBehaviorModels)
SP_CLASS_ATTR(DisplayFlagsData,showForceFields)
SP_CLASS_ATTR(DisplayFlagsData,showInteractionForceFields)
SP_CLASS_ATTR(DisplayFlagsData,showCollision)
SP_CLASS_ATTR(DisplayFlagsData,showCollisionModels)
SP_CLASS_ATTR(DisplayFlagsData,showBoundingCollisionModels)
SP_CLASS_ATTR(DisplayFlagsData,showMapping)
SP_CLASS_ATTR(DisplayFlagsData,showMappings)
SP_CLASS_ATTR(DisplayFlagsData,showMechanicalMappings)
SP_CLASS_ATTR(DisplayFlagsData,showOptions)
SP_CLASS_ATTR(DisplayFlagsData,showWireFrame)
SP_CLASS_ATTR(DisplayFlagsData,showNormals)
SP_CLASS_ATTRS_END

SP_CLASS_TYPE_PTR_ATTR(DisplayFlagsData,Data<DisplayFlags>,Data)

