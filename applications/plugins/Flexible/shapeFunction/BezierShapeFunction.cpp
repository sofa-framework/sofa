/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define FLEXIBLE_BezierShapeFunction_CPP

#include <Flexible/config.h>
#include "../shapeFunction/BezierShapeFunction.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace shapefunction
{

using namespace core::behavior;

SOFA_DECL_CLASS(BezierShapeFunction)

// Register in the Factory
int BezierShapeFunctionClass = core::RegisterObject("Computes Bezier shape functions")
#ifndef SOFA_FLOAT
        .add< BezierShapeFunction<ShapeFunctiond> >(true)
//        .add< BezierShapeFunction<ShapeFunction2d> >()
#endif
#ifndef SOFA_DOUBLE
        .add< BezierShapeFunction<ShapeFunctionf> >()
//        .add< BezierShapeFunction<ShapeFunction2f> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_Flexible_API BezierShapeFunction<ShapeFunctiond>;
//template class SOFA_Flexible_API BezierShapeFunction<ShapeFunction2d>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Flexible_API BezierShapeFunction<ShapeFunctionf>;
//template class SOFA_Flexible_API BezierShapeFunction<ShapeFunction2f>;
#endif

}
}
}
