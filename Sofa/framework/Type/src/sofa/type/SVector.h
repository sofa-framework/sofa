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

#include <sofa/type/vector.h>

namespace sofa::type
{

//======================================================================
/// Same as type::vector, + delimitors on serialization
//======================================================================
template<class T>
class SVector: public type::vector<T, type::CPUMemoryManager<T> >
{
public:
    using Inherit = type::vector<T, type::CPUMemoryManager<T> >;

    typedef type::CPUMemoryManager<T>  Alloc;
    /// Size
    typedef typename Inherit::Size Size;
    /// reference to a value (read-write)
    //typedef typename Inherit::reference reference;
    /// const reference to a value (read only)
    //typedef typename Inherit::const_reference const_reference;

    /// Basic onstructor
    SVector() : Inherit() {}
    /// Constructor
    SVector(Size n, const T& value): Inherit(n,value) {}
    /// Constructor
    SVector(int n, const T& value): Inherit(n,value) {}
    /// Constructor
    SVector(long n, const T& value): Inherit(n,value) {}
    /// Constructor
    explicit SVector(Size n): Inherit(n) {}
    /// Constructor
    SVector(const Inherit& x): Inherit(x) {}
    /// Move constructor
    SVector(Inherit&& v): Inherit(std::move(v)) {}


    /// Copy operator
    SVector<T>& operator=(const Inherit& x)
    {
        Inherit::operator=(x);
        return *this;
    }
    /// Move assignment operator
    SVector<T>& operator=(Inherit&& v)
    {
        Inherit::operator=(std::move(v));
        return *this;
    }

    /// Constructor
    SVector(typename SVector<T>::const_iterator first, typename SVector<T>::const_iterator last): Inherit(first,last) {}

    std::ostream& write ( std::ostream& os ) const
    {
        if ( !this->empty() )
        {
            typename SVector<T>::const_iterator i = this->begin();
            os << "[ " << *i;
            ++i;
            for ( ; i!=this->end(); ++i )
                os << ", " << *i;
            os << " ]";

        }
        else os << "[]"; // empty vector
        return os;
    }

    std::istream& read ( std::istream& in )
    {
        T t;
        this->clear();
        char c;

        in >> c;

        if( in.eof() ) return in; // empty stream

        if ( c != '[' )
        {
            std::cerr << "Error (SVector) " << "read : Bad begin character : " << c << ", expected  [" << std::endl;
            in.setstate(std::ios::failbit);
            return in;
        }
        const std::streampos pos = in.tellg();
        in >> c;
        if( c == ']' ) // empty vector
        {
            return in;
        }
        else
        {
            in.seekg( pos ); // coming-back to previous character
            c = ',';
            while( !in.eof() && c == ',')
            {
                in >> t;
                this->push_back ( t );
                in >> c;
            }
            if ( c != ']' )
            {
                std::cerr << "Error (SVector) " << "read : Bad end character : " << c << ", expected  ]" << std::endl;
                in.setstate(std::ios::failbit);
                return in;
            }
        }
        return in;
    }

/// Output stream
    inline friend std::ostream& operator<< ( std::ostream& os, const SVector<T>& vec )
    {
        return vec.write(os);
    }

/// Input stream
    inline friend std::istream& operator>> ( std::istream& in, SVector<T>& vec )
    {
        return vec.read(in);
    }

};

/// reading specialization for std::string
/// SVector begins by [, ends by ] and separates elements with ,
/// string elements must be delimited by ' or " (like a list of strings in python).
/// example: ['string1' ,  "string 2 ",'etc...' ]
template<>
std::istream& SVector<std::string>::read( std::istream& in );
template<>
std::ostream& SVector<std::string>::write( std::ostream& os ) const;


} // namespace sofa::type
