#ifndef SOFA_HELPER_PAIR_H
#define SOFA_HELPER_PAIR_H

#include <sofa/helper/helper.h>

#include <utility>
#include <iostream>
#include <string>


/// adding string serialization to std::pair to make it compatible with Data
/// \todo: refactoring of the containers required
/// More info PR #113: https://github.com/sofa-framework/sofa/pull/113


namespace std
{

/// Output stream
template<class T1, class T2>
std::ostream& operator<< ( std::ostream& o, const std::pair<T1,T2>& p )
{
    return o << "[" << p.first << ", " << p.second << "]";
}

template<class T1, class T2>
std::istream& readDelimiter ( std::istream& in, std::pair<T1,T2>& p )
{
    char c;
    std::streampos pos = in.tellg();
    in >> c;
    if( c == ']' ) // empty pair
        return in;
    else {
        in.seekg( pos ); // coming-back to previous character
        if (!(in >> p.first)) {
            msg_error("pair") << "Error reading [,] separated values";
            in.setstate(std::ios::failbit);
            return in;
        }
        if (!(in >> c)) {
            msg_error("pair") << "Error reading [,] separated values";
            in.setstate(std::ios::failbit);
            return in;
        }
        if (c!=',') {
            msg_error("pair") << "read: bad separating character: " << c << ", expected  ,";
            in.setstate(std::ios::failbit);
            return in;
        }
        if (!(in >> p.second)) {
            msg_error("pair") << "Error reading [,] separated values";
            in.setstate(std::ios::failbit);
            return in;
        }
        if (!(in >> c)) {
            msg_error("pair") << "Error reading [,] separated values";
            in.setstate(std::ios::failbit);
            return in;
        }
        if (c!=']') {
            msg_error("pair") << "read: bad end character: " << c << ", expected ]";
            in.setstate(std::ios::failbit);
            return in;
        }
        return in;
    }
}

template<class T1, class T2>
std::istream& read( std::istream& in, std::pair<T1,T2>& p )
{
    in >> p.first >> p.second;
    if (in.eof())
        in.clear(std::ios::eofbit);
    if (in.fail())
        msg_error("Pair") << "Error reading space separated values";
    return in;
}

/// Input stream
template<class T1, class T2>
std::istream& operator>> ( std::istream& in, std::pair<T1,T2>& p )
{
    std::streampos pos = in.tellg();
    char c;
    in >> c;
    if( in.eof() )
        return in; // empty stream
    if ( c == '[' ) {
        return readDelimiter(in, p);
    }
    else {
        in.seekg( pos ); // coming-back to the previous position
        return read(in, p);
    }
}


} // namespace std

#endif
