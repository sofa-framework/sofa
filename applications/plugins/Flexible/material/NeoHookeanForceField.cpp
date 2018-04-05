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
#define SOFA_NeoHookeanFORCEFIELD_CPP

#include <Flexible/config.h>
#include "../material/NeoHookeanForceField.h"
#include "../types/StrainTypes.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

SOFA_DECL_CLASS(NeoHookeanForceField);

using namespace defaulttype;

// Register in the Factory
int NeoHookeanForceFieldClass = core::RegisterObject("NeoHookean's Law for isotropic homogeneous materials")

        .add< NeoHookeanForceField< I331Types > >(true)
        .add< NeoHookeanForceField< U331Types > >()
        .add< NeoHookeanForceField< U321Types > >()
        ;

template class SOFA_Flexible_API NeoHookeanForceField< I331Types >;
template class SOFA_Flexible_API NeoHookeanForceField< U331Types >;
template class SOFA_Flexible_API NeoHookeanForceField< U321Types >;

}
}
}

