/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "ManualMapping.h"
#include <sofa/core/Plugin.h>

#include "ManualLinearMapping.inl"
#include <sofa/defaulttype/VecTypes.h>

using sofa::component::mapping::ManualLinearMapping;
using namespace sofa::defaulttype;


class ManualMappingPlugin: public sofa::core::Plugin {
public:
    ManualMappingPlugin(): Plugin("ManualMapping") {
        setDescription("Quick implementation of mapping where everything is manually given.");
        setVersion("0.1");
        setLicense("LGPL");

#ifdef SOFA_FLOAT
        addComponent< ManualLinearMapping< Vec3fTypes, Vec3fTypes > >();
#else
        addComponent< ManualLinearMapping< Vec3dTypes, Vec3dTypes > >();
#endif
        setDescription("ManualLinearMapping", "Maps displacement from subspace with basis given by J.");

#if !defined(SOFA_DOUBLE) && !defined(SOFA_FLOAT)
        addTemplateInstance< ManualLinearMapping< Vec3fTypes, Vec3fTypes > >();
        addTemplateInstance< ManualLinearMapping< Vec3fTypes, Vec3dTypes > >();
        addTemplateInstance< ManualLinearMapping< Vec3dTypes, Vec3fTypes > >();
#endif
    }
};

SOFA_PLUGIN(ManualMappingPlugin);


#ifdef SOFA_FLOAT
template class SOFA_ManualMapping_API ManualLinearMapping< Vec3fTypes, Vec3fTypes >;
#else
template class SOFA_ManualMapping_API ManualLinearMapping< Vec3dTypes, Vec3dTypes >;
#endif

#if !defined(SOFA_DOUBLE) && !defined(SOFA_FLOAT)
template class SOFA_ManualMapping_API ManualLinearMapping< Vec3fTypes, Vec3fTypes >;
template class SOFA_ManualMapping_API ManualLinearMapping< Vec3dTypes, Vec3fTypes >;
template class SOFA_ManualMapping_API ManualLinearMapping< Vec3fTypes, Vec3dTypes >;
#endif
