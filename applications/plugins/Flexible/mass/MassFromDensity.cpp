/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_Flexible_MassFromDensity_CPP

#include <Flexible/config.h>
#include "MassFromDensity.h"
#include <sofa/core/ObjectFactory.h>
#include "../types/AffineTypes.h"

namespace sofa {
namespace component {
namespace engine {

using namespace defaulttype;

SOFA_DECL_CLASS(MassFromDensity)

int MassFromDensityClass = core::RegisterObject("Compute a mass matrix from a density image")
        .add<MassFromDensity<Affine3Types,ImageD > >(true)
        .add<MassFromDensity<Affine3Types,ImageF > >()
        .add<MassFromDensity<Affine3Types,ImageUI > >()
        .add<MassFromDensity<Affine3Types,ImageUC > >()
;

template class SOFA_Flexible_API MassFromDensity<Affine3Types,ImageD  >;
template class SOFA_Flexible_API MassFromDensity<Affine3Types,ImageF  >;
template class SOFA_Flexible_API MassFromDensity<Affine3Types,ImageUI >;
template class SOFA_Flexible_API MassFromDensity<Affine3Types,ImageUC >;

}
}
}
