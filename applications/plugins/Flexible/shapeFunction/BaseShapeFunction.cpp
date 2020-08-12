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
#define FLEXIBLE_BaseShapeFunction_CPP

#include <Flexible/config.h>

#include <sofa/defaulttype/TemplatesAliases.h>

#include "../shapeFunction/BaseShapeFunction.h"


namespace sofa
{
namespace core
{
namespace behavior
{

template class SOFA_Flexible_API BaseShapeFunction<ShapeFunction3>;
template class SOFA_Flexible_API BaseShapeFunction<ShapeFunction2>;

}
}

namespace defaulttype {
RegisterTemplateAlias ShapeFunctionAlias0("ShapeFunctiond", core::behavior::ShapeFunction3d::Name(), true);
RegisterTemplateAlias ShapeFunctionAlias1("ShapeFunction3", core::behavior::ShapeFunction3::Name() );
RegisterTemplateAlias ShapeFunctionAlias2("ShapeFunction2", core::behavior::ShapeFunction2::Name() );
}

}
