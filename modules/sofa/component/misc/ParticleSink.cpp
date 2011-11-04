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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "ParticleSink.h"
#include "sofa/core/behavior/Constraint.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include "sofa/defaulttype/Vec3Types.h"

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(ParticleSink)

int ParticleSinkClass = core::RegisterObject("Parametrable particle generator")
#ifndef SOFA_FLOAT
        .add< ParticleSink<defaulttype::Vec3dTypes> >()
        .add< ParticleSink<defaulttype::Vec2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ParticleSink<defaulttype::Vec3fTypes> >()
        .add< ParticleSink<defaulttype::Vec2fTypes> >()
#endif
        ;

}
}
}
