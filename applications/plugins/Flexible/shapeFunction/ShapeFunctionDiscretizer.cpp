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
#define FLEXIBLE_ShapeFunctionDiscretizer_CPP

#include <Flexible/config.h>
#include "../shapeFunction/ShapeFunctionDiscretizer.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(ShapeFunctionDiscretizer)

// Register in the Factory
int ShapeFunctionDiscretizerClass = core::RegisterObject("Discretize shape functions in an image")
        .add< ShapeFunctionDiscretizer<ImageUC> >(true)
        .add< ShapeFunctionDiscretizer<ImageD> >()
        .add< ShapeFunctionDiscretizer<ImageB> >()
        .add< ShapeFunctionDiscretizer<ImageF> >()
        ;

template class SOFA_Flexible_API ShapeFunctionDiscretizer<ImageUC>;
template class SOFA_Flexible_API ShapeFunctionDiscretizer<ImageD>;
template class SOFA_Flexible_API ShapeFunctionDiscretizer<ImageB>;
template class SOFA_Flexible_API ShapeFunctionDiscretizer<ImageF>;

}
}
}
