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
#define SOFA_HELPER_VECTOR_STRING_DEFINITION
#include <sofa/type/vector_String.h>
#include <sofa/type/vector_T.inl>

#include <iostream>
#include <sstream>


/// All integral types are considered as extern templates.
namespace sofa::type
{

/// Output stream
/// Specialization for writing vectors of unsigned char
template<>
SOFA_TYPE_API std::ostream& vector<std::string>::write(std::ostream& os) const
{
    std::string separator = "";
    os << "[";
    for(auto& v : (*this))
    {
        os << separator << '"' << v << '"';
        separator = ", ";
    }
    os << "]";
    return os;
}

/// Input stream
/// Parses the format produced by write(): ["elem1", "elem2", ...]
/// Falls back to plain whitespace-separated tokens if no leading '[' is found.
/// Sets stream to fail state and writes to std::cerr on bad formatting.
template<>
SOFA_TYPE_API std::istream& vector<std::string>::read( std::istream& in )
{
    this->clear();

    // Skip leading whitespace
    in >> std::ws;

    if (in.eof())
        return in; // empty input is valid

    char c = static_cast<char>(in.peek());

    if (c == '[')
    {
        // --- Bracketed quoted format: ["elem1", "elem2", ...] ---
        in.get(); // consume '['

        in >> std::ws;
        if (in.peek() == ']')
        {
            in.get(); // consume ']', empty array
            return in;
        }

        while (true)
        {
            in >> std::ws;

            if (!in.good())
            {
                std::cerr << "vector<string>::read: unexpected end of stream, expected '\"' or ']'" << std::endl;
                in.setstate(std::ios_base::failbit);
                return in;
            }

            char next = static_cast<char>(in.peek());

            if (next == ']')
            {
                in.get(); // consume ']'
                break;
            }

            if (next != '"')
            {
                std::cerr << "vector<string>::read: expected '\"' but got '" << next << "'" << std::endl;
                in.setstate(std::ios_base::failbit);
                return in;
            }

            in.get(); // consume opening '"'

            std::string token;
            bool closed = false;
            char ch;
            while (in.get(ch))
            {
                if (ch == '"')
                {
                    closed = true;
                    break;
                }
                token += ch;
            }

            if (!closed)
            {
                std::cerr << "vector<string>::read: unterminated quoted string" << std::endl;
                in.setstate(std::ios_base::failbit);
                return in;
            }

            this->push_back(token);

            // After the closing quote: expect ',' or ']'
            in >> std::ws;
            if (!in.good())
            {
                std::cerr << "vector<string>::read: unexpected end of stream after element" << std::endl;
                in.setstate(std::ios_base::failbit);
                return in;
            }

            char sep = static_cast<char>(in.peek());
            if (sep == ',')
            {
                in.get(); // consume ','
            }
            else if (sep == ']')
            {
                // will be consumed on next loop iteration
            }
            else
            {
                std::cerr << "vector<string>::read: expected ',' or ']' but got '" << sep << "'" << std::endl;
                in.setstate(std::ios_base::failbit);
                return in;
            }
        }
    }
    else
    {
        // --- Plain whitespace-separated fallback ---
        std::string token;
        while (in >> token)
        {
            this->push_back(token);
        }
        if (in.rdstate() & std::ios_base::eofbit)
        {
            in.clear();
        }
    }

    return in;
}


} // namespace sofa::type

template class SOFA_TYPE_API sofa::type::vector<std::string>;
