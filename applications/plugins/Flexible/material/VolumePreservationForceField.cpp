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
#define SOFA_VolumePreservationFORCEFIELD_CPP

#include <Flexible/config.h>
#include "../material/VolumePreservationForceField.h"
#include "../types/StrainTypes.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

SOFA_DECL_CLASS(VolumePreservationForceField);

using namespace defaulttype;

// Register in the Factory
int VolumePreservationForceFieldClass = core::RegisterObject("volume Preservation law for isotropic homogeneous materials")

        .add< VolumePreservationForceField< I331Types > >(true)
//.add< VolumePreservationForceField< I332Types > >()
//.add< VolumePreservationForceField< I333Types > >()
        .add< VolumePreservationForceField< U331Types > >()
        ;

template class SOFA_Flexible_API VolumePreservationForceField< I331Types >;
//template class SOFA_Flexible_API VolumePreservationForceField< I332Types >;
//template class SOFA_Flexible_API VolumePreservationForceField< I333Types >;
template class SOFA_Flexible_API VolumePreservationForceField< U331Types >;

}
}
}

