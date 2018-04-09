/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_HELPER_VECTOR_CPP
#include <sofa/helper/vector.h>
#include <sofa/helper/vector_device.h>
#include <sofa/helper/integer_id.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <cassert>
#include <iostream>

namespace sofa
{

namespace helper
{

#ifdef DEBUG_OUT_VECTOR
int cptid = 0;
#endif

void SOFA_HELPER_API vector_access_failure(const void* vec, unsigned size, unsigned i, const std::type_info& type)
{
    msg_error("vector") << "in vector<"<<gettypename(type)<<"> " << std::hex << (long)vec << std::dec << " size " << size << " : invalid index " << (int)i;
    BackTrace::dump();
    assert(i < size);
}

void SOFA_HELPER_API vector_access_failure(const void* vec, unsigned size, unsigned i, const std::type_info& type, const char* tindex)
{
    msg_error("vector") << "in vector<"<<gettypename(type)<<", integer_id<"<<tindex<<"> > " << std::hex << (long)vec << std::dec << " size " << size << " : invalid index " << (int)i;
    BackTrace::dump();
    assert(i < size);
}

/// Convert the string 's' into an unsigned int. The error are reported in msg & numErrors
/// is incremented.
int SOFA_HELPER_API getInteger(const std::string& s, std::stringstream& msg, unsigned int& numErrors)
{
    const char* attrstr=s.c_str();
    char* end=nullptr;
    int retval = strtol(attrstr, &end, 10);

    /// It is important to check that the string was totally parsed to report
    /// message to users because a silent error is the worse thing that can happen in UX.
    if(end ==  attrstr+strlen(attrstr))
        return retval ;

    if(numErrors<5)
        msg << "    - problem while parsing '" << s <<"' as Integer'. Replaced by 0 instead." << msgendl ;
    if(numErrors==5)
        msg << "   - ... " << msgendl;
    numErrors++ ;
    return 0 ;
}



/// Convert the string 's' into an unsigned int. The error are reported in msg & numErrors
/// is incremented.
unsigned int SOFA_HELPER_API getUnsignedInteger(const std::string& s, std::stringstream& msg, unsigned int& numErrors)
{
    const char* attrstr=s.c_str();
    char* end=nullptr;

    long long tmp = strtoll(attrstr, &end, 10);

    /// If there is minus sign we exit.
    if( tmp<0 ){
        if(numErrors<5)
            msg << "   - problem while parsing '" << s <<"' as Unsigned Integer because the minus sign is not allowed'. Replaced by 0 instead." << msgendl ;
        if(numErrors==5)
            msg << "   - ... " << msgendl;
        numErrors++ ;
        return 0 ;
    }

    /// It is important to check that the string was totally parsed to report
    /// message to users because a silent error is the worse thing that can happen in UX.
    if(end !=  attrstr+strlen(attrstr))
    {
        if(numErrors<5)
            msg << "   - problem while parsing '" << s <<"' as Unsigned Integer'. Replaced by 0 instead." << msgendl ;
        if(numErrors==5)
            msg << "   - ... " << msgendl;
        numErrors++ ;
        return 0 ;
    }

    return (unsigned int)tmp ;
}



} // namespace helper
} // namespace sofa
