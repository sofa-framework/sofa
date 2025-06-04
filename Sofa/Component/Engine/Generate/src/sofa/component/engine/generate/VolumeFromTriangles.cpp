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
*                           Plugin SoftRobots                                 *
*                                                                             *
* This plugin is also distributed under the GNU LGPL (Lesser General          *
* Public License) license with the same conditions than SOFA.                 *
*                                                                             *
* Contributors: Defrost team  (INRIA, University of Lille, CNRS,              *
*               Ecole Centrale de Lille)                                      *
*                                                                             *
* Contact information: https://project.inria.fr/softrobot/contact/            *
******************************************************************************/
#include <sofa/component/engine/generate/VolumeFromTriangles.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::engine::generate
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

void registerVolumeFromTriangles(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("This class computes the volume of a given closed surfacic mesh.")
                            .add<  VolumeFromTriangles<Vec3Types> >(true)
                            .add<  VolumeFromTriangles<Rigid3Types> >());
}

template class SOFA_COMPONENT_ENGINE_GENERATE_API VolumeFromTriangles<Vec3Types>;
template class SOFA_COMPONENT_ENGINE_GENERATE_API VolumeFromTriangles<Rigid3Types>;

}

