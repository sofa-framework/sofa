/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* This component is open-source                                               *
*                                                                             *
* Authors: Bruno Carrez                                                       *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*****************************************************************************
* User of this library should read the documentation
* in the messaging.h file.
******************************************************************************/

#include <sstream>
using std::ostringstream ;

#include <iostream>
using std::endl ;
using std::cout ;
using std::cerr ;

#include "DefaultStyleMessageFormatter.h"


namespace sofa
{

namespace helper
{

namespace messaging
{

static DefaultStyleMessageFormatter s_DefaultStyleMessageFormatter;


/*
string SofaStyleMessageHandler::format(const Message& m)
{
    ostringstream out;
    format(m, out);
    return out.str();
}

string operator*(const string& s, unsigned int num)
{
    string tmp;
    for(unsigned int i=0;i<num;i++){
        tmp+=s ;
    }
    return tmp ;
}

*/
#define BLUE "\033[1;34m "
#define GREEN "\033[1;32m "
#define CYAN "\033[1;36m "
#define RED "\033[1;31m "
#define PURPLE "\033[1;35m "
#define YELLOW "\033[1;33m "
#define WHITE "\033[1;37m "
#define ENDL " \033[0m"

MessageFormatter* DefaultStyleMessageFormatter::getInstance()
{
    return &s_DefaultStyleMessageFormatter;
}


void reformat(unsigned int begin, const std::string& input, std::ostream& out)
{
    unsigned int linebreak = 120 ;
    unsigned int idx=begin ;
    unsigned int curr=0 ;
    while(curr < input.size()){
        if(idx==linebreak){
            out << endl ;
            for(unsigned int i=0;i<begin;i++)
                out << ' ' ;
            idx=begin ;

            if(input[curr]==' ')curr ++ ;
        }
        if(curr >= input.size()) break;
        out << input[curr++] ;
        idx++;
    }
}

void DefaultStyleMessageFormatter::formatMessage(const Message& m,std::ostream& out)
{
    std::ostringstream tmpStr;
    if(m.type() == "info"){
        tmpStr << GREEN << "[INFO]" << ENDL ;
    }else if(m.type() == "warn"){
        tmpStr << CYAN << "[WARN]" << ENDL ;
    }else if(m.type() == "error"){
        tmpStr << RED << "[ERROR]" << ENDL ;
    }else if(m.type() == "fatal"){
        tmpStr << RED << "[FATAL]" << ENDL ;
    }

    tmpStr << "[" << m.sendername() << "]: ";

    //todo(damien): this is ugly !! the -11 is to remove the color codes from the string !
    // fix this by making a function that count the size of a string ignoring the escapes...
    // or adding the color code when the formatting is already finished.
    unsigned int numspaces = tmpStr.str().size() - 10 ;
    if(numspaces >= tmpStr.str().size() ){
        numspaces = tmpStr.str().size() ;
    }
    reformat(numspaces+2, m.message(), tmpStr) ;
    out << tmpStr.str();
}


} // messaging
} // helper
} // sofa
