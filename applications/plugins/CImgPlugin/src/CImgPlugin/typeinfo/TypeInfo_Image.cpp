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
#include <sofa/defaulttype/typeinfo/TypeInfo_Bool.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Integer.h>
#include <CImgPlugin/typeinfo/TypeInfo_Image.h>
using sofa::defaulttype::loadInRepository;
using sofa::defaulttype::TypeInfoType;
using sofa::defaulttype::TypeInfoRegistryTools;

namespace
{

int registerTypeInfos(const std::string& target)
{
    loadInRepository<sofa::defaulttype::ImageC>(target);
    loadInRepository<sofa::defaulttype::ImageUC>(target);
    loadInRepository<sofa::defaulttype::ImageI>(target);
    loadInRepository<sofa::defaulttype::ImageUI>(target);
    loadInRepository<sofa::defaulttype::ImageS>(target);
    loadInRepository<sofa::defaulttype::ImageUS>(target);
    loadInRepository<sofa::defaulttype::ImageL>(target);
    loadInRepository<sofa::defaulttype::ImageUL>(target);
    loadInRepository<sofa::defaulttype::ImageF>(target);
    loadInRepository<sofa::defaulttype::ImageD>(target);
    loadInRepository<sofa::defaulttype::ImageB>(target);

    TypeInfoRegistryTools::dumpRegistryContentToStream(std::cout, TypeInfoType::NAMEONLY, target);
    TypeInfoRegistryTools::dumpRegistryContentToStream(std::cout, TypeInfoType::COMPLETE, target);
    return 1;
}

static int inited = registerTypeInfos(sofa_tostring(SOFA_TARGET));

}
