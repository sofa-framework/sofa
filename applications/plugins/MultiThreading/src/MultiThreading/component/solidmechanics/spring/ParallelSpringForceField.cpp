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
#include <MultiThreading/component/solidmechanics/spring/ParallelSpringForceField.inl>
#include <sofa/core/ObjectFactory.h>

namespace multithreading::component::solidmechanics::spring
{

void registerParallelSpringForceField(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Parallel stiff springs.")
                             .add< ParallelSpringForceField<sofa::defaulttype::Vec3Types> >()
                             .add< ParallelSpringForceField<sofa::defaulttype::Vec2Types> >()
                             .add< ParallelSpringForceField<sofa::defaulttype::Vec1Types> >()
                             .add< ParallelSpringForceField<sofa::defaulttype::Vec6Types> >()
                             .add< ParallelSpringForceField<sofa::defaulttype::Rigid3Types> >());
}

template class SOFA_MULTITHREADING_PLUGIN_API ParallelSpringForceField<sofa::defaulttype::Vec3Types>;
template class SOFA_MULTITHREADING_PLUGIN_API ParallelSpringForceField<sofa::defaulttype::Vec2Types>;
template class SOFA_MULTITHREADING_PLUGIN_API ParallelSpringForceField<sofa::defaulttype::Vec1Types>;
template class SOFA_MULTITHREADING_PLUGIN_API ParallelSpringForceField<sofa::defaulttype::Vec6Types>;
template class SOFA_MULTITHREADING_PLUGIN_API ParallelSpringForceField<sofa::defaulttype::Rigid3Types>;

}
