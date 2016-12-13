#include "SVector.h"

namespace sofa
{

namespace helper
{

/// reading specialization for std::string
/// SVector begins by [, ends by ] and separates elements with ,
/// string elements must be delimited by ' or " (like a list of strings in python).
///
/// Note this is a quick&dirty implementation and it could be improved
template<>
SOFA_HELPER_API std::istream& SVector<std::string>::read( std::istream& in )
{
    this->clear();

    std::string s = std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());

    size_t f = s.find_first_of('[');
    if( f == std::string::npos )
    {
        // a '[' must be present
        msg_error("SVector") << "read : a '[' is expected as beginning marker.";
        return in;
    }
    else
    {
        std::size_t f2 = s.find_first_not_of(' ',f);
        if( f2!=std::string::npos && f2 < f )
        {
            // the '[' must be the first character
            msg_error("SVector") << "read : Bad begin character, expected [";
            return in;
        }
    }

    size_t e = s.find_last_of(']');
    if( e == std::string::npos )
    {
        // a ']' must be present
        msg_error("SVector") << "read : a ']' is expected as ending marker.";
        return in;
    }
    else
    {
        // the ']' must be the last character
        std::size_t e2 = s.find_last_not_of(' ');
        if( e2!=std::string::npos && e2 > e )
        {
            msg_error("SVector") << "read : Bad end character, expected ]";
            return in;
        }
    }


    // looking for elements in between '[' and ']' separated by ','
    while(f<e-1)
    {
        size_t i = s.find_first_of(',', f+1); // i is the ',' position after the previous ','

        if( i == std::string::npos ) // no more ',' => last element
            i=e;


        std::size_t f2 = s.find_first_of("\"'",f+1);
        if( f2==std::string::npos )
        {
            msg_error("SVector") << "read : Bad begin string character, expected \" or '";
            this->clear();
            return in;
        }

        std::size_t i2 = s.find_last_of(s[f2],i-1);
        if( i2==std::string::npos )
        {
            msg_error("SVector") << "read : Bad end string character, expected "<<s[f2];
            this->clear();
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
SOFA_HELPER_API std::ostream& SVector<std::string>::write( std::ostream& os ) const
{
    if ( !this->empty() )
    {
        SVector<std::string>::const_iterator i = this->begin(), iend=this->end();
        os << "[ '" << *i <<"'";
        ++i;
        for ( ; i!=iend; ++i )
            os << " , '" << *i <<"'";
        os << " ]";
    }
    else os << "[]"; // empty vector
    return os;
}


}
}
