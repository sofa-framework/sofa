/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
//========================================================
// Yet another command line parser.
// Francois Faure, iMAGIS-GRAVIR, May 2001
//========================================================
#include "ArgumentParser.h"
#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace helper
{

typedef std::istringstream istrstream;

ArgumentBase::ArgumentBase(char s, string l, string h, bool m)
    : shortName(s)
    , longName(l)
    , help(h)
    , mandatory(m)
    , isSet(false)
{}

/// Base destructor: does nothing.
ArgumentBase::~ArgumentBase()
{}

/// print short name, long name, help
void ArgumentBase::print () const
{
    std::cout << "-" << shortName <<",\t--"<< longName <<":\t" << help;
    if( mandatory ) std::cout<< " (required) ";
    std::cout << "  (default: ";  printValue();
    std::cout << ")" << std::endl;
}


//========================================================================
/// Constructor using a global help string
ArgumentParser::ArgumentParser( const string& helpstr, char hlpShrt, const string& hlpLng )
    : files(NULL)
    , globalHelp( helpstr )
    , helpShortName(hlpShrt)
    , helpLongName(hlpLng)
{}

/// Constructor using a global help string and a list of filenames
ArgumentParser::ArgumentParser( std::vector<std::string>* files, const string& helpstr, char hlpShrt, const string& hlpLng )
    : files(files)
    , globalHelp( helpstr )
    , helpShortName(hlpShrt)
    , helpLongName(hlpLng)
{}

/// Constructor using a global help string
ArgumentParser::~ArgumentParser()
{
    for( ArgVec::const_iterator a=commands.begin(), aend=commands.end(); a!=aend; ++a )
        delete (*a);
}

/** Parse a command line
\param argc number of arguments + 1, as usual in C
\param argv arguments
*/
void ArgumentParser::operator () ( int argc, char** argv )
{
    std::list<std::string> str;
    for (int i=1; i<argc; ++i)
        str.push_back(std::string(argv[i]));
    (*this)(str);
}

void ArgumentParser::operator () ( std::list<std::string> str )
{
    string shHelp("-");  shHelp.push_back( helpShortName );
    string lgHelp("--"); lgHelp.append( helpLongName );
    string name;
    while( !str.empty() )
    {
        name = str.front();
        str.pop_front();
//		std::cout << "name = " << name << std::endl;
//		std::cout << "lgHelp = " << lgHelp << std::endl;
//		std::cout << "shHelp = " << shHelp << std::endl;

        // display help
        if( name == shHelp || name == lgHelp )
        {
            if( globalHelp.size()>0 ) std::cout<< globalHelp <<std::endl;
            std::cout << "(short name, long name, description, default value)\n-h,\t--help: this help" << std::endl;
            std::cout << std::boolalpha;
            for( ArgVec::const_iterator a=commands.begin(), aend=commands.end(); a!=aend; ++a )
                (*a)->print();
            std::cout << std::noboolalpha;
            if( files )
                std::cout << "others: file names" << std::endl;
            exit(EXIT_FAILURE);
        }

        // not an option
        else if( files && name[0]!='-' )
        {
            files->push_back(name);
        }

        // indicating the next argument is not an option
        else if( files && name=="--" )
        {
            files->push_back(str.front());
            str.pop_front();
        }

        // long name
        else if( name.length() > 1 && name[0]=='-' && name[1]=='-' )
        {
            string a;
            for( unsigned int i=2; i<name.length(); ++i )
            {
                a += name[i];
            }
            if( longName.find(a) != longName.end() )
            {
                if( !(longName[ a ]->read( str )))
                    msg_warning("ArgumentParser") << "Could not read value for option: " << name;
                else parameter_set[longName[ a ]] = true;
            }
            else
                msg_warning("ArgumentParser") << "Unknown option: " << name;
        }

        // short names (possibly concatenated)
        else if( name.length() > 1 && name[0]=='-' && name[1]!='-' )
        {
            for( unsigned int i=1; i<name.length(); ++i )
            {
                char a = name[i];
                if( shortName.find(a) != shortName.end() )
                {
                    if( !(shortName[ a ]->read( str )))
                        msg_warning("ArgumentParser") << "Could not read value for option: " << name;

                    else parameter_set[shortName[ a ]] = true;
                }
                else
                    msg_warning("ArgumentParser") << "Unknown option: " << name;
            }
        }

        else
            msg_warning("ArgumentParser") << "Unknown option: " << name;

    }

    // Unset mandatory arguments ?
    bool unset = false;
    for( ArgVec::const_iterator cm = commands.begin(), cmend=commands.end(); cm != cmend; ++cm )
    {
        if( (*cm)->mandatory && !(*cm)->isSet )
        {
            if( !unset )
            {
                std::cout << "Please set the following parameters: (short name, long name, description)" << std::endl;
                unset = true;
            }
            (*cm)->print();
        }
    }
    if( unset ) exit(EXIT_FAILURE);
}

} // namespace helper

} // namespace sofa
