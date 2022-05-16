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
#pragma once

#include <sofa/defaulttype/typeinfo/models/IncompleteTypeInfo.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Scalar.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_FixedArray.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::defaulttype
{

template<sofa::Size N, typename real>
struct DataTypeInfo< sofa::defaulttype::StdRigidTypes<N, real> > : public IncompleteTypeInfo<sofa::defaulttype::StdRigidTypes<N, real>>
{
    static std::string GetTypeName() { std::ostringstream o; o << "RigidTypes<" << N << "," << DataTypeInfo<real>::GetTypeName() << ">"; return o.str(); }
    static std::string name()
    {
        std::ostringstream o;
        o << "RigidTypes" << N << DataTypeInfo<real>::name();
        return o.str();
    }
};

template<sofa::Size N, typename real>
struct DataTypeInfo< sofa::defaulttype::RigidCoord<N,real> > : public FixedArrayTypeInfo< sofa::defaulttype::RigidCoord<N,real>, sofa::defaulttype::RigidCoord<N,real>::total_size >
//struct DataTypeInfo< sofa::defaulttype::RigidCoord<N,real> > : public IncompleteTypeInfo< sofa::defaulttype::RigidCoord<N,real> >
{
    static std::string GetTypeName() { std::ostringstream o; o << "RigidCoord<" << N << "," << DataTypeInfo<real>::GetTypeName() << ">"; return o.str(); }
    static std::string name()
    {
        std::ostringstream o;
        o << "RigidCoord" << N << DataTypeInfo<real>::name() ;
        return o.str();
    }
};

template<sofa::Size N, typename real>
struct DataTypeInfo< sofa::defaulttype::RigidDeriv<N,real> > : public FixedArrayTypeInfo< sofa::defaulttype::RigidDeriv<N,real>,
                                                                                          sofa::defaulttype::RigidDeriv<N,real>::total_size >
//struct DataTypeInfo< sofa::defaulttype::RigidDeriv<N,real> > : public IncompleteTypeInfo< sofa::defaulttype::RigidDeriv<N,real> >
{
    static std::string GetTypeName() { std::ostringstream o; o << "RigidDeriv<" << N << "," << DataTypeInfo<real>::GetTypeName() << ">"; return o.str(); }
    static std::string name()
    {
        std::ostringstream o;
        o << "RigidDeriv" << N << DataTypeInfo<real>::name();
        return o.str();
    }
};

template<sofa::Size N, typename real>
struct DataTypeInfo< sofa::defaulttype::RigidMass<N,real> > : public IncompleteTypeInfo<RigidMass<N,real>>
{
    static std::string GetTypeName() { std::ostringstream o; o << "RigidMass<" << N << "," << DataTypeInfo<real>::GetTypeName() << ">"; return o.str(); }
    static std::string name()
    {
        std::ostringstream o;
        o << "RigidMass" << N << DataTypeInfo<real>::name();
        return o.str();
    }
};

} /// namespace sofa::defaulttype

