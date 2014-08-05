/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/
#ifndef NAMETYPES_H_
#define NAMETYPES_H_

#include <string>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/vector.h>
#include <Topology/generic/dart.h>
#include <Eigen/Eigen>
namespace CGoGN
{

/**
 * get name used for save/load of attribute type
 */
template <typename T>
std::string nameOfType(const T& v)
{
	return v.CGoGNnameOfType();
}

template <> inline std::string nameOfType(const Eigen::Matrix3d& /*v*/) { return "EigenMat3d"; }
template <> inline std::string nameOfType(const Eigen::Matrix4d& /*v*/) { return "EigenMat4d"; }

template <> inline std::string nameOfType(const sofa::defaulttype::Mat<24,6,float>& /*v*/) { return "Mat<24,6,float>"; }
template <> inline std::string nameOfType(const sofa::defaulttype::Mat<24,6,double>& /*v*/) { return "Mat<24,6,double>"; }
template <> inline std::string nameOfType(const sofa::defaulttype::Mat<24,24,float>& /*v*/) { return "Mat<24,24,float>"; }
template <> inline std::string nameOfType(const sofa::defaulttype::Mat<24,24,double>& /*v*/) { return "Mat<24,24,double>"; }
template <> inline std::string nameOfType(const sofa::helper::fixed_array<sofa::defaulttype::Vec<3,float>,8>& /*v*/) { return "fixed_array<sVec<3,float>,8>f"; }
template <> inline std::string nameOfType(const sofa::helper::fixed_array<sofa::defaulttype::Vec<3,double>,8>& /*v*/) { return "fixed_array<sVec<3,double>,8>f"; }


template <> inline std::string nameOfType(const bool& /*v*/) { return "bool"; }
template <> inline std::string nameOfType(const sofa::defaulttype::Mat4x4f& /*v*/) { return "Mat4x4f"; }
template <> inline std::string nameOfType(const sofa::defaulttype::Mat4x4d& /*v*/) { return "Mat4x4d"; }
template <> inline std::string nameOfType(const sofa::helper::fixed_array<unsigned int, 4>& /*v*/) { return "sofa4UnsignedArray"; }
template <> inline std::string nameOfType(const sofa::helper::fixed_array<unsigned int, 8>& /*v*/) { return "sofa8UnsignedArray"; }
template <> inline std::string nameOfType(const std::vector<SReal> & /*v*/) { return "SRealVector"; }
template <> inline std::string nameOfType(const std::vector<Dart> & /*v*/) { return "dartVector"; }
template <> inline std::string nameOfType(const sofa::helper::vector<unsigned int>& /*v*/) { return "sofaUnsignedHelperVector"; }
template <> inline std::string nameOfType(const sofa::helper::vector<sofa::defaulttype::Vector3> & /*v*/) { return "sofavec3HelperVector"; }
template <> inline std::string nameOfType(const sofa::defaulttype::Vec2f& /*v*/) { return "sofaVec2f"; }
template <> inline std::string nameOfType(const sofa::defaulttype::Vec2d& /*v*/) { return "sofaVec2d"; }
template <> inline std::string nameOfType(const sofa::defaulttype::Vec3f& /*v*/) { return "sofaVec3f"; }
template <> inline std::string nameOfType(const sofa::defaulttype::Vec3d& /*v*/) { return "sofaVec3d"; }
template <> inline std::string nameOfType(const sofa::defaulttype::Vec4f& /*v*/) { return "sofaVec4f"; }
template <> inline std::string nameOfType(const sofa::defaulttype::Vec4d& /*v*/) { return "sofaVec4d"; }

template <> inline std::string nameOfType(const char& /*v*/) { return "char"; }

template <> inline std::string nameOfType(const short& /*v*/) { return "short"; }

template <> inline std::string nameOfType(const int& /*v*/) { return "int"; }

template <> inline std::string nameOfType(const long& /*v*/) { return "long"; }

template <> inline std::string nameOfType(const long long& /*v*/) { return "long long"; }

template <> inline std::string nameOfType(const unsigned char& /*v*/) { return "unsigned char"; }

template <> inline std::string nameOfType(const unsigned short& /*v*/) { return "unsigned short"; }

template <> inline std::string nameOfType(const unsigned int& /*v*/) { return "unsigned int"; }

template <> inline std::string nameOfType(const unsigned long& /*v*/) { return "unsigned long"; }

template <> inline std::string nameOfType(const unsigned long long& /*v*/) { return "unsigned long long"; }

template <> inline std::string nameOfType(const float& /*v*/) { return "float"; }

template <> inline std::string nameOfType(const double& /*v*/) { return "double"; }

template <> inline std::string nameOfType(const std::string& /*v*/) { return "std::string"; }

// Ply compatibility
template <typename T>
std::string nameOfTypePly(const T& v)
{
	return v.CGoGNnameOfType();
}

template <> inline std::string nameOfTypePly(const char& /*v*/) { return "int8"; }

template <> inline std::string nameOfTypePly(const short int& /*v*/) { return "int16"; }

template <> inline std::string nameOfTypePly(const int& /*v*/) { return "int32"; }

template <> inline std::string nameOfTypePly(const long int& /*v*/) { return "invalid"; }

template <> inline std::string nameOfTypePly(const unsigned char& /*v*/) { return "uint8"; }

template <> inline std::string nameOfTypePly(const unsigned short int& /*v*/) { return "uint16"; }

template <> inline std::string nameOfTypePly(const unsigned int& /*v*/) { return "uint32"; }

template <> inline std::string nameOfTypePly(const unsigned long int& /*v*/) { return "invalid"; }

template <> inline std::string nameOfTypePly(const float& /*v*/) { return "float32"; }

template <> inline std::string nameOfTypePly(const double& /*v*/) { return "float64"; }

template <> inline std::string nameOfTypePly(const std::string& /*v*/) { return "invalid"; }

}

#endif /* NAMETYPES_H_ */
