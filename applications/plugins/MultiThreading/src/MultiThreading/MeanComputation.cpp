/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <MultiThreading/MeanComputation.inl>

#include <sofa/core/ObjectFactory.h>

#include <sofa/type/Vec.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::engine
{

int MeanComputationEngineClass = core::RegisterObject("Compute the mean of the input elements")
    .add< MeanComputation<defaulttype::Vec3Types> >(true) // default template
    .add< MeanComputation<defaulttype::Vec1Types> >()
    .add< MeanComputation<defaulttype::Vec2Types> >()
    .add< MeanComputation<defaulttype::Rigid2Types> >()
    .add< MeanComputation<defaulttype::Rigid3Types> >()
    ;

template class SOFA_MULTITHREADING_PLUGIN_API MeanComputation< defaulttype::Vec3Types >;
template class SOFA_MULTITHREADING_PLUGIN_API MeanComputation< defaulttype::Vec1Types >;
template class SOFA_MULTITHREADING_PLUGIN_API MeanComputation< defaulttype::Vec2Types >;
template class SOFA_MULTITHREADING_PLUGIN_API MeanComputation< defaulttype::Rigid2Types >;
template class SOFA_MULTITHREADING_PLUGIN_API MeanComputation< defaulttype::Rigid3Types >;

} // namespace sofa::component::engine
