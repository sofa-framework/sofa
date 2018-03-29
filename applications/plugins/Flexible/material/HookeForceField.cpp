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
#define SOFA_HookeFORCEFIELD_CPP

#include <Flexible/config.h>
#include "HookeForceField.h"
#include "../types/StrainTypes.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

using namespace defaulttype;

SOFA_DECL_CLASS(HookeForceField)

// Register in the Factory
int HookeForceFieldClass = core::RegisterObject("Hooke's Law for isotropic homogeneous materials")

        .add< HookeForceField< E331Types > >(true)
        .add< HookeForceField< E321Types > >()
        .add< HookeForceField< E311Types > >()
        .add< HookeForceField< E332Types > >()
        .add< HookeForceField< E333Types > >()
        .add< HookeForceField< E221Types > >()

//        .add< HookeForceField< D331Types > >()
//        .add< HookeForceField< D321Types > >()
//        .add< HookeForceField< D332Types > >()

        .add< HookeForceField< U331Types > >()
        .add< HookeForceField< U321Types > >()
        ;

template class SOFA_Flexible_API HookeForceField< E331Types >;
template class SOFA_Flexible_API HookeForceField< E321Types >;
template class SOFA_Flexible_API HookeForceField< E311Types >;
template class SOFA_Flexible_API HookeForceField< E332Types >;
template class SOFA_Flexible_API HookeForceField< E333Types >;
template class SOFA_Flexible_API HookeForceField< E221Types >;

//template class SOFA_Flexible_API HookeForceField< D331Types >;
//template class SOFA_Flexible_API HookeForceField< D321Types >;
//template class SOFA_Flexible_API HookeForceField< D332Types >;

template class SOFA_Flexible_API HookeForceField< U331Types >;
template class SOFA_Flexible_API HookeForceField< U321Types >;





SOFA_DECL_CLASS(HookeOrthotropicForceField)

// Register in the Factory
int HookeOrthotropicForceFieldClass = core::RegisterObject("Hooke's Law for Orthotropic homogeneous materials")

        .add< HookeOrthotropicForceField< E331Types > >(true)
        .add< HookeOrthotropicForceField< E321Types > >()
        .add< HookeOrthotropicForceField< E332Types > >()
        .add< HookeOrthotropicForceField< E333Types > >()
        ;

template class SOFA_Flexible_API HookeOrthotropicForceField< E331Types >;
template class SOFA_Flexible_API HookeOrthotropicForceField< E321Types >;
template class SOFA_Flexible_API HookeOrthotropicForceField< E332Types >;
template class SOFA_Flexible_API HookeOrthotropicForceField< E333Types >;




SOFA_DECL_CLASS(HookeTransverseForceField)

// Register in the Factory
int HookeTransverseForceFieldClass = core::RegisterObject("Hooke's Law for Transversely isotropic homogeneous materials (symmetry about X axis)")

        .add< HookeTransverseForceField< E331Types > >(true)
        .add< HookeTransverseForceField< E332Types > >()
        .add< HookeTransverseForceField< E333Types > >()
        ;

template class SOFA_Flexible_API HookeTransverseForceField< E331Types >;
template class SOFA_Flexible_API HookeTransverseForceField< E332Types >;
template class SOFA_Flexible_API HookeTransverseForceField< E333Types >;

}
}
}

