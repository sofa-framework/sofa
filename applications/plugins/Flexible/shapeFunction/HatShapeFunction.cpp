/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#define FLEXIBLE_HatShapeFunction_CPP

#include <Flexible/config.h>
#include "../shapeFunction/HatShapeFunction.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace shapefunction
{

SOFA_DECL_CLASS(HatShapeFunction)

// Register in the Factory
int HatShapeFunctionClass = core::RegisterObject("Computes compactly supported hat shape functions")

        .add< HatShapeFunction<sofa::core::behavior::ShapeFunction> >(true)
        ;

template class SOFA_Flexible_API HatShapeFunction<sofa::core::behavior::ShapeFunction>;
}
}
}
