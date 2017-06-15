/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#define SOFA_HEMLStVKFORCEFIELD_CPP

#include <Flexible/config.h>
#include "HEMLStVKForceField.h"
#include "../types/StrainTypes.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

using namespace defaulttype;

SOFA_DECL_CLASS(HEMLStVKForceField)

// Register in the Factory
int HEMLStVKForceFieldClass = core::RegisterObject("St Venant-Kirchhoff (Hooke's Law for isotropic homogeneous materials)")
        .add< HEMLStVKForceField< Vec1Types > >(true)
        ;

template class SOFA_Flexible_API HEMLStVKForceField< Vec1Types >;



}
}
}

