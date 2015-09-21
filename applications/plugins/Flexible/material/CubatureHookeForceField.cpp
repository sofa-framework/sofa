/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#define SOFA_CUBATUREHookeFORCEFIELD_CPP

#include "../initFlexible.h"
#include "CubatureHookeForceField.h"
#include "../types/StrainTypes.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

using namespace defaulttype;

SOFA_DECL_CLASS(CubatureHookeForceField)

// Register in the Factory
int CubatureHookeForceFieldClass = core::RegisterObject("Hooke's Law for isotropic homogeneous materials")

        .add< CubatureHookeForceField< E331Types > >(true)
        .add< CubatureHookeForceField< E321Types > >()
        .add< CubatureHookeForceField< E311Types > >()
        .add< CubatureHookeForceField< E332Types > >()
        .add< CubatureHookeForceField< E333Types > >()
        .add< CubatureHookeForceField< E221Types > >()

//        .add< CubatureHookeForceField< D331Types > >()
//        .add< CubatureHookeForceField< D321Types > >()
//        .add< CubatureHookeForceField< D332Types > >()

        .add< CubatureHookeForceField< U331Types > >()
        .add< CubatureHookeForceField< U321Types > >()
        ;

template class SOFA_Flexible_API CubatureHookeForceField< E331Types >;
template class SOFA_Flexible_API CubatureHookeForceField< E321Types >;
template class SOFA_Flexible_API CubatureHookeForceField< E311Types >;
template class SOFA_Flexible_API CubatureHookeForceField< E332Types >;
template class SOFA_Flexible_API CubatureHookeForceField< E333Types >;
template class SOFA_Flexible_API CubatureHookeForceField< E221Types >;

//template class SOFA_Flexible_API CubatureHookeForceField< D331Types >;
//template class SOFA_Flexible_API CubatureHookeForceField< D321Types >;
//template class SOFA_Flexible_API CubatureHookeForceField< D332Types >;

template class SOFA_Flexible_API CubatureHookeForceField< U331Types >;
template class SOFA_Flexible_API CubatureHookeForceField< U321Types >;

}
}
}

