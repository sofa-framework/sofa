/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_DEFAULTTYPE_DATATYPEINFO_H
#define SOFA_DEFAULTTYPE_DATATYPEINFO_H

namespace sofa
{

namespace defaulttype
{

template<class DataType>
class DataTypeInfo
{
public:
    static unsigned int size() { return DataType::size(); }

    template <typename T>
    static void getValue(const DataType &type, unsigned int index, T& value)
    {
        value = static_cast<T>(type[index]);
    }

    template<typename T>
    static void setValue(DataType &type, unsigned int index, const T& value )
    {
        type[index] = static_cast<typename DataType::value_type>(value);
    }
};

template<>
class DataTypeInfo<double>
{
public:
    static unsigned int size() { return 1; }

    template <typename T>
    static void getValue(const double &type, unsigned int /* index */, T& value)
    {
        value = static_cast<T>(type);
    }

    template<typename T>
    static void setValue(double &type, unsigned int /* index */, const T& value )
    {
        type = static_cast<double>(value);
    }
};

template<>
class DataTypeInfo<float>
{
public:
    static unsigned int size() { return 1; }
    static double getValue(const float &type, unsigned int /* index */) { return type; }

    template <typename T>
    static void getValue(const float &type, unsigned int /* index */, T& value)
    {
        value = static_cast<T>(type);
    }

    template<typename T>
    static void setValue(float &type, unsigned int /* index */, const T& value )
    {
        type = static_cast<float>(value);
    }
};

} // namespace defaulttype

} // namespace sofa

#endif  // SOFA_DEFAULTTYPE_DATATYPEINFO_H
