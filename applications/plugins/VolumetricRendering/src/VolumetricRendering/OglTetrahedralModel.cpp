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
#define SOFA_COMPONENT_VISUALMODEL_OGLTETRAHEDRALMODEL_CPP

#include <VolumetricRendering/OglTetrahedralModel.inl>

#include <sofa/core/ObjectFactory.h>

namespace volumetricrendering
{

using namespace sofa::defaulttype;

void registerOglTetrahedralModel(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Tetrahedral model for OpenGL display.")
    .add< OglTetrahedralModel<Vec3Types> >());
}

template class SOFA_VOLUMETRICRENDERING_API OglTetrahedralModel<Vec3Types>;

} // namespace volumetricrendering
