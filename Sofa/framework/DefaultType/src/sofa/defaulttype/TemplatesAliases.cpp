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
#include <sofa/defaulttype/TemplatesAliases.h>

#include <iostream>
#include <map>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>

namespace sofa::defaulttype
{

typedef std::map<std::string, TemplateAlias> TemplateAliasesMap;
typedef TemplateAliasesMap::const_iterator TemplateAliasesMapIterator;
TemplateAliasesMap& getTemplateAliasesMap()
{
	static TemplateAliasesMap theMap;
	return theMap;
}

bool TemplateAliases::addAlias(const std::string& name, const std::string& result, const bool doWarnUser)
{
    TemplateAliasesMap& templateAliases = getTemplateAliasesMap();
    if (templateAliases.find(name) != templateAliases.end())
    {
        msg_warning("ObjectFactory") << "cannot create template alias " << name <<
            " as it already exists";
        return false;
    }
    else
    {
        templateAliases[name] = std::make_pair(result, doWarnUser);
        return true;
    }
}

const TemplateAlias* TemplateAliases::getTemplateAlias(const std::string &name)
{
    TemplateAliasesMap& templateAliases = getTemplateAliasesMap();
    const TemplateAliasesMapIterator it = templateAliases.find(name);
    if (it != templateAliases.end())
        return  &(it->second);
    return nullptr;
}

std::string TemplateAliases::resolveAlias(const std::string& name)
{
	TemplateAliasesMap& templateAliases = getTemplateAliasesMap();
	TemplateAliasesMapIterator it = templateAliases.find(name);
	if (it != templateAliases.end())
        return it->second.first;
	else if (name.find(",") != std::string::npos) // Multiple templates, resolve each one
	{
		std::string resolved = name;
		std::string::size_type first = 0;
		while (true)
		{
			std::string::size_type last = resolved.find_first_of(",", first);
			if (last == std::string::npos) // Take until the end of the string if there is no more comma
				last = resolved.size();
			std::string token = resolved.substr(first, last-first);

			// Replace the token with the alias (if there is one)
			it = templateAliases.find(token);
			if (it != templateAliases.end())
                resolved.replace(first, last-first, it->second.first);

			// Recompute the start of next token as we can have changed the length of the string
			first = resolved.find_first_of(",", first);
			if (first == std::string::npos)
				break;
			++first;
		}

		return resolved;
	}
	else
		return name;
}

RegisterTemplateAlias::RegisterTemplateAlias(const std::string& alias, const std::string& result, const bool doWarnUser)
{
    TemplateAliases::addAlias(alias, result, doWarnUser);
}


/// The following types are the generic 'precision'
static RegisterTemplateAlias Vec1Alias("Vec1", sofa::defaulttype::Vec1Types::Name());
static RegisterTemplateAlias Vec2Alias("Vec2", sofa::defaulttype::Vec2Types::Name());
static RegisterTemplateAlias Vec3Alias("Vec3", sofa::defaulttype::Vec3Types::Name());
static RegisterTemplateAlias Vec6Alias("Vec6", sofa::defaulttype::Vec6Types::Name());
static RegisterTemplateAlias Rigid2Alias("Rigid2", sofa::defaulttype::Rigid2Types::Name());
static RegisterTemplateAlias Rigid3Alias("Rigid3", sofa::defaulttype::Rigid3Types::Name());
static RegisterTemplateAlias CompressedRowSparseMatrixAlias("CompressedRowSparseMatrix", sofa::linearalgebra::CompressedRowSparseMatrix<SReal>::Name());
static RegisterTemplateAlias CompressedRowSparseMatrixMat3x3Alias("CompressedRowSparseMatrixMat3x3", sofa::linearalgebra::CompressedRowSparseMatrix<type::Mat<3, 3, SReal>>::Name());
static RegisterTemplateAlias Mat2x2Alias("Mat2x2", sofa::defaulttype::DataTypeName<type::Mat<2, 2, SReal>>::name());
static RegisterTemplateAlias Mat3x3Alias("Mat3x3", sofa::defaulttype::DataTypeName<type::Mat<3, 3, SReal>>::name());
static RegisterTemplateAlias Mat4x4Alias("Mat4x4", sofa::defaulttype::DataTypeName<type::Mat<4, 4, SReal>>::name());
static RegisterTemplateAlias Mat6x6Alias("Mat6x6", sofa::defaulttype::DataTypeName<type::Mat<6, 6, SReal>>::name());

/// Compatibility aliases for niceness.
static RegisterTemplateAlias RigidAlias("Rigid", sofa::defaulttype::Rigid3Types::Name(), true);

static RegisterTemplateAlias Rigid2fAlias("Rigid2f", sofa::defaulttype::Rigid2Types::Name(), isSRealDouble());
static RegisterTemplateAlias Rigid3fAlias("Rigid3f", sofa::defaulttype::Rigid3Types::Name(), isSRealDouble());
static RegisterTemplateAlias Vec1fAlias("Vec1f", sofa::defaulttype::Vec1Types::Name(), isSRealDouble());
static RegisterTemplateAlias Vec2fAlias("Vec2f", sofa::defaulttype::Vec2Types::Name(), isSRealDouble());
static RegisterTemplateAlias Vec3fAlias("Vec3f", sofa::defaulttype::Vec3Types::Name(), isSRealDouble());
static RegisterTemplateAlias Vec6fAlias("Vec6f", sofa::defaulttype::Vec6Types::Name(), isSRealDouble());

static RegisterTemplateAlias Vec1dAlias("Vec1d", sofa::defaulttype::Vec1Types::Name(), isSRealFloat());
static RegisterTemplateAlias Vec2dAlias("Vec2d", sofa::defaulttype::Vec2Types::Name(), isSRealFloat());
static RegisterTemplateAlias Vec3dAlias("Vec3d", sofa::defaulttype::Vec3Types::Name(), isSRealFloat());
static RegisterTemplateAlias Vec6dAlias("Vec6d", sofa::defaulttype::Vec6Types::Name(), isSRealFloat());
static RegisterTemplateAlias Rigid2dAlias("Rigid2d", sofa::defaulttype::Rigid2Types::Name(), isSRealFloat());
static RegisterTemplateAlias Rigid3dAlias("Rigid3d", sofa::defaulttype::Rigid3Types::Name(), isSRealFloat());

// Compatibility aliases used previously in DataExchange (see PR#3380)
static RegisterTemplateAlias floatAlias("float", sofa::defaulttype::DataTypeName<float>::name(), true);
static RegisterTemplateAlias doubleAlias("double", sofa::defaulttype::DataTypeName<double>::name(), true);
static RegisterTemplateAlias vector_intAlias("vector<int>", sofa::defaulttype::DataTypeName<sofa::type::vector<int> >::name(), true);
static RegisterTemplateAlias vector_uintAlias("vector<unsigned_int>", sofa::defaulttype::DataTypeName<sofa::type::vector<unsigned int> >::name(), true);
static RegisterTemplateAlias vector_floatAlias("vector<float>", sofa::defaulttype::DataTypeName<sofa::type::vector<float> >::name(), true);
static RegisterTemplateAlias vector_doubleAlias("vector<double>", sofa::defaulttype::DataTypeName<sofa::type::vector<double> >::name(), true);

// Compatibility aliases used previously (see PR#3465)
static RegisterTemplateAlias intAlias("int", sofa::defaulttype::DataTypeName<int>::name(), true);
static RegisterTemplateAlias dataIntAlias("Data<int>", sofa::defaulttype::DataTypeName<sofa::type::vector<int>>::name(), true);
static RegisterTemplateAlias dataDoubleAlias("Data<double>", sofa::defaulttype::DataTypeName<sofa::type::vector<double>>::name(), true);
static RegisterTemplateAlias dataBoolAlias("Data<bool>", sofa::defaulttype::DataTypeName<sofa::type::vector<bool>>::name(), true);
static RegisterTemplateAlias dataVec2uAlias("Data<Vec<2u,unsigned int>>", sofa::defaulttype::DataTypeName<type::vector<type::Vec2u>>::name(), true);
static RegisterTemplateAlias dataVec2dAlias("Data<Vec<2u,double>>", sofa::defaulttype::DataTypeName<sofa::type::vector<type::Vec2d>>::name(), true);
static RegisterTemplateAlias dataVec3dAlias("Data<Vec<3u,double>>", sofa::defaulttype::DataTypeName<sofa::type::vector<type::Vec3d>>::name(), true);
static RegisterTemplateAlias dataVec4dAlias("Data<Vec<4u,double>>", sofa::defaulttype::DataTypeName<sofa::type::vector<type::Vec4d>>::name(), true);
static RegisterTemplateAlias dataRigidCoord2dAlias("Data<RigidCoord<2u,double>>", sofa::defaulttype::DataTypeName<defaulttype::Rigid2Types::VecCoord>::name(), true);
static RegisterTemplateAlias dataRigidDeriv2dAlias("Data<RigidDeriv<2u,double>>", sofa::defaulttype::DataTypeName<defaulttype::Rigid2Types::VecDeriv>::name(), true);
static RegisterTemplateAlias dataRigidCoord3dAlias("Data<RigidCoord<3u,double>>", sofa::defaulttype::DataTypeName<defaulttype::Rigid3Types::VecCoord>::name(), true);
static RegisterTemplateAlias dataRigidDeriv3dAlias("Data<RigidDeriv<3u,double>>", sofa::defaulttype::DataTypeName<defaulttype::Rigid3Types::VecDeriv>::name(), true);

} // namespace sofa::defaulttype
