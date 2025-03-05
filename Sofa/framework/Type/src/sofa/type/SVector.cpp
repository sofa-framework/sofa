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
#include <sofa/type/SVector.h>

namespace sofa::type
{

/// reading specialization for std::string
/// SVector begins by [, ends by ] and separates elements with ,
/// string elements must be delimited by ' or " (like a list of strings in python).
///
/// Note this is a quick&dirty implementation and it could be improved
template<>
SOFA_TYPE_API std::istream& SVector<std::string>::read( std::istream& in )
{
    this->clear();

    const std::string s = std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());

    size_t f = s.find_first_of('[');
    if( f == std::string::npos )
    {
        // a '[' must be present
        std::cerr << "Error (SVector) " << "read : a '[' is expected as beginning marker." << std::endl;
        in.setstate(std::ios::failbit);
        return in;
    }
    else
    {
        const std::size_t f2 = s.find_first_not_of(' ',f);
        if( f2!=std::string::npos && f2 < f )
        {
            // the '[' must be the first character
            std::cerr << "Error (SVector) " << "read : Bad begin character, expected [" << std::endl;
            in.setstate(std::ios::failbit);
            return in;
        }
    }

    const size_t e = s.find_last_of(']');
    if( e == std::string::npos )
    {
        // a ']' must be present
        std::cerr << "Error (SVector) " << "read : a ']' is expected as ending marker." << std::endl;
        in.setstate(std::ios::failbit);
        return in;
    }
    else
    {
        // the ']' must be the last character
        const std::size_t e2 = s.find_last_not_of(' ');
        if( e2!=std::string::npos && e2 > e )
        {
            std::cerr << "Error (SVector) " << "read : Bad end character, expected ]" << std::endl;
            in.setstate(std::ios::failbit);
            return in;
        }
    }


    // looking for elements in between '[' and ']' separated by ','
    while(f<e-1)
    {
        size_t i = s.find_first_of(',', f+1); // i is the ',' position after the previous ','

        if( i == std::string::npos ) // no more ',' => last element
            i=e;


        const std::size_t f2 = s.find_first_of("\"'",f+1);
        if( f2==std::string::npos )
        {
            std::cerr << "Error (SVector) " << "read : Bad begin string character, expected \" or '" << std::endl;
            this->clear();
            in.setstate(std::ios::failbit);
            return in;
        }

        const std::size_t i2 = s.find_last_of(s[f2],i-1);
        if( i2==std::string::npos )
        {
            std::cerr << "Error (SVector) " << "read : Bad end string character, expected "<<s[f2] << std::endl;
            this->clear();
            in.setstate(std::ios::failbit);
            return in;
        }


        if( i2-f2-1<=0 ) // empty string
            this->push_back( "" );
        else
            this->push_back( s.substr(f2+1,i2-f2-1) );

        f=i; // the next element will begin after the ','
    }


    return in;
}

template<>
SOFA_TYPE_API std::ostream& SVector<std::string>::write( std::ostream& os ) const
{
    if ( !this->empty() )
    {
        auto i = this->begin();
        const auto iend = this->end();
        os << "[ '" << *i <<"'";
        ++i;
        for ( ; i!=iend; ++i )
            os << " , '" << *i <<"'";
        os << " ]";
    }
    else os << "[]"; // empty vector
    return os;
}

} // namespace sofa::type
