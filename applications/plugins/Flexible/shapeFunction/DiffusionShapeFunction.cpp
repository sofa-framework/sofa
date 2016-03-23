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
#define FLEXIBLE_DiffusionShapeFunction_CPP

#include <Flexible/config.h>
#include "../shapeFunction/DiffusionShapeFunction.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace shapefunction
{

using namespace defaulttype;
using namespace core::behavior;

SOFA_DECL_CLASS(DiffusionShapeFunction)

// Register in the Factory
int DiffusionShapeFunctionClass = core::RegisterObject("Computes shape functions based on diffusion in images")

        .add< DiffusionShapeFunction<ShapeFunction,ImageUC> >()
        .add< DiffusionShapeFunction<ShapeFunction,ImageD> >(true)
        .add< DiffusionShapeFunction<ShapeFunction,ImageF> >()
        .add< DiffusionShapeFunction<ShapeFunction,ImageB> >()
        ;

template class SOFA_Flexible_API DiffusionShapeFunction<ShapeFunction,ImageUC>;
template class SOFA_Flexible_API DiffusionShapeFunction<ShapeFunction,ImageD>;
template class SOFA_Flexible_API DiffusionShapeFunction<ShapeFunction,ImageF>;
template class SOFA_Flexible_API DiffusionShapeFunction<ShapeFunction,ImageB>;

}
}
}
