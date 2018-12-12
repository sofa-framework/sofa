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
#define SOFA_COMPONENT_ENGINE_HAUSDORFFDISTANCE_CPP
#include "HausdorffDistance.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::defaulttype;

int HausdorffDistanceClass = core::RegisterObject("Compute the Hausdorff distance of two point clouds")
        .add< HausdorffDistance<Vec1Types> >()
        .add< HausdorffDistance<Vec2Types> >()
        .add< HausdorffDistance<Vec3Types> >(true)
        .add< HausdorffDistance<Rigid2Types> >()
        .add< HausdorffDistance<Rigid3Types> >()
 //SOFA_FLOAT
        ;

template class SOFA_GENERAL_ENGINE_API HausdorffDistance<Vec1Types>;
template class SOFA_GENERAL_ENGINE_API HausdorffDistance<Vec2Types>;
template class SOFA_GENERAL_ENGINE_API HausdorffDistance<Vec3Types>;
template class SOFA_GENERAL_ENGINE_API HausdorffDistance<Rigid2Types>;
template class SOFA_GENERAL_ENGINE_API HausdorffDistance<Rigid3Types>;
 //SOFA_FLOAT


} //
} // namespace component

} // namespace sofa

