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
#define SOFA_CORE_DATATYPES_DATAMATSYM_DEFINITION
#include <sofa/core/datatype/Data[MatSym].h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[Scalar].h>

#include <sofa/core/objectmodel/Data.inl>
#include <sofa/defaulttype/typeinfo/models/IncompleteTypeInfo.h>

namespace sofa::defaulttype
{
template<int L, typename REAL>
struct DataTypeInfo< sofa::defaulttype::MatSym<L,REAL> > : public IncompleteTypeInfo<sofa::defaulttype::MatSym<L,REAL>>
{
    static std::string GetTypeName()
    {
        std::ostringstream o;
        o << "MatSym<" << L << "," << DataTypeInfo<REAL>::GetTypeName() << ">";
        return o.str();
    }

    static std::string name()
    {
        std::ostringstream o;
        o << "MatSym" << L << DataTypeInfo<REAL>::name();
        return o.str();
    }
};
}


namespace sofa::core::objectmodel
{
template class Data<sofa::defaulttype::MatSym<3,float>>;
template class Data<sofa::defaulttype::MatSym<3,double>>;
}
