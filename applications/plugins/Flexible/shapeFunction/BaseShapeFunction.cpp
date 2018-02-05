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

#ifndef SOFA_FLOAT
template class SOFA_Flexible_API BaseShapeFunction<ShapeFunctiond>;
template class SOFA_Flexible_API BaseShapeFunction<ShapeFunction2d>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Flexible_API BaseShapeFunction<ShapeFunctionf>;
template class SOFA_Flexible_API BaseShapeFunction<ShapeFunction2f>;
#endif


}
}

namespace defaulttype {
#ifndef SOFA_FLOAT
RegisterTemplateAlias ShapeFunctionAlias("ShapeFunction", "ShapeFunctiond");
RegisterTemplateAlias ShapeFunctionAlias2("ShapeFunction2", "ShapeFunction2d");
#else
RegisterTemplateAlias ShapeFunctionAlias("ShapeFunction", "ShapeFunctionf");
RegisterTemplateAlias ShapeFunctionAlias2("ShapeFunction2", "ShapeFunction2f");
#endif

}

}
