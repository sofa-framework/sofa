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
#include <Flexible/types/typeinfo/TypeInfo_AffineTypes.h>

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
#include <sofa/defaulttype/TypeInfoRegistryTools.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Scalar.h>
#include <Flexible/types/typeinfo/TypeInfo_AffineTypes.h>
#include <Flexible/types/typeinfo/TypeInfo_QuadraticTypes.h>
#include <Flexible/types/typeinfo/TypeInfo_DeformationGradientTypes.h>
#include <Flexible/types/typeinfo/TypeInfo_StrainTypes.h>
using sofa::defaulttype::loadInRepository;
using sofa::defaulttype::TypeInfoType;
using sofa::defaulttype::TypeInfoRegistryTools;
using namespace sofa::defaulttype;

namespace
{

int registerTypeInfos(const std::string& target)
{
    loadInRepository<Affine3dTypes::Coord>(target);
    loadVectorForType<Affine3dTypes::Coord>(target);

    loadInRepository<Quadratic3dTypes::Coord>(target);
    loadVectorForType<Quadratic3dTypes::Coord>(target);

    loadInRepository<E331fTypes::Coord>(target);
    loadInRepository<E321fTypes::Coord>(target);
    loadInRepository<E311fTypes::Coord>(target);
    loadInRepository<E332fTypes::Coord>(target);
    loadInRepository<E333fTypes::Coord>(target);
    loadInRepository<E221fTypes::Coord>(target);

    loadInRepository<E331dTypes::Coord>(target);
    loadInRepository<E321dTypes::Coord>(target);
    loadInRepository<E311dTypes::Coord>(target);
    loadInRepository<E332dTypes::Coord>(target);
    loadInRepository<E333dTypes::Coord>(target);
    loadInRepository<E221dTypes::Coord>(target);

    loadInRepository<F331dTypes::Coord>(target);
    loadInRepository<F332dTypes::Coord>(target);
    loadInRepository<F321dTypes::Coord>(target);
    loadInRepository<F311dTypes::Coord>(target);
    loadInRepository<F221dTypes::Coord>(target);

    loadVectorForType<F331dTypes::Coord>(target);
    loadVectorForType<F332dTypes::Coord>(target);
    loadVectorForType<F321dTypes::Coord>(target);
    loadVectorForType<F311dTypes::Coord>(target);
    loadVectorForType<F221dTypes::Coord>(target);


    loadInRepository<F331dTypes::Coord>(target);
    loadInRepository<F332dTypes::Coord>(target);
    loadInRepository<F321dTypes::Coord>(target);
    loadInRepository<F311dTypes::Coord>(target);
    loadInRepository<F221dTypes::Coord>(target);

    loadVectorForType<F331dTypes::Coord>(target);
    loadVectorForType<F332dTypes::Coord>(target);
    loadVectorForType<F321dTypes::Coord>(target);
    loadVectorForType<F311dTypes::Coord>(target);
    loadVectorForType<F221dTypes::Coord>(target);


    TypeInfoRegistryTools::dumpRegistryContentToStream(std::cout, TypeInfoType::NAMEONLY, target);
    TypeInfoRegistryTools::dumpRegistryContentToStream(std::cout, TypeInfoType::COMPLETE, target);
    return 1;
}

static int inited = registerTypeInfos(sofa_tostring(SOFA_TARGET));

}
