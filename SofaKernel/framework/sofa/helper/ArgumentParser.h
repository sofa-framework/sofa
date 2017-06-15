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

#ifndef SOFA_HELPER_ARGUMENTPARSER_H
#define SOFA_HELPER_ARGUMENTPARSER_H

#include <sofa/helper/helper.h>
#include <sofa/helper/logging/Messaging.h>

#include <iostream>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <list>
#include <vector>

namespace sofa
{

namespace helper
{

typedef std::istringstream istrstream;

/// Abstract base class for all command line arguments
class SOFA_HELPER_API ArgumentBase
{
public:
    /// character string
    typedef std::string string;

    /** Constructor
    \param s short name
    \param l long name
    \param h help
    \param m true iff the argument is mandatory
    */
    ArgumentBase(char s, string l, string h, bool m);

    /// Base destructor: does nothing.
    virtual ~ArgumentBase();

    /// Read the command line
    virtual bool read( std::list<std::string>& str ) = 0;

    /// Print the value of the associated variable
    virtual void printValue() const =0;

    /// print short name, long name, help
    void print () const;

    char shortName; ///< Short name
    string longName;  ///< Long name
    string help;      ///< Help message

    /// True iff the value must be set
    bool mandatory;

    /// True iff a value has bee read on the command line
    bool isSet;

};


//=========================================================================

/** Command line argument.
\brief Contains a pointer to a value which can be parsed by a ArgumentParser.

Contains also a short name, a long name and a help message.

@see ArgumentParser
*/
template < class T = void* >
class Argument : public ArgumentBase
{
public:

    /** Constructor
    \param t a pointer to the value
    \param sn short name of the argument
    \param ln long name of the argument
    \param h help on the argument
    \param m true iff the argument is mandatory
    */
    Argument( T* t, char sn, string ln, string h, bool m )
        : ArgumentBase(sn,ln,h,m)
        , ptr(t)
    {}

    inline void printValue() const ;

private:
    /// Pointer to the parameter
    T* ptr;


    /** Try to read argument value from an input stream.
        Return false if failed
    */
    inline bool read( std::list<std::string>& str )
    {
        if (str.empty()) return false;
        std::string s = str.front();
        str.pop_front();
        istrstream istr( s.c_str() );
        if( ! (istr >> *ptr) ) return false;
        else
        {
            isSet = true;
            return true;
        }
    }

};

/** Specialization for reading lists.
Lists are used for options that can be repeted
Example: run -D ELEM1 -D ELEM2 ...
*/
/*
template<class TE> inline
bool Argument< std::vector<TE> >::read( std::list<std::string>& str)
{
    if (str.empty()) return false;
    std::string s = str.front();
    str.pop_front();
    TE val;
    istrstream istr( s.c_str() );
    if( ! (istr >> val) ) return false;
    else {
        isSet = true;
        *ptr.push_back(vak);
        return true;
    }
}
*/

template<> inline
bool Argument< std::vector< std::string > >::read( std::list<std::string>& str)
{
    if (str.empty()) return false;
    std::string s = str.front();
    str.pop_front();
    isSet = true;
    ptr->push_back(s);
    return true;
}

/** Specialization for flag reading booleans.
Booleans are seen as flags that you can set to TRUE using the command line.
Example: run --verbose
The advantage is that you do not have to set the value, it is automatically TRUE.
The drawback is that reading a boolean necessarily sets it to TRUE. Currently you can not set a boolean to FALSE using this parser.
*/
template<> inline
bool Argument<bool>::read( std::list<std::string>& )
{
    *ptr = true;
    isSet = true;
    return true;
}

template<> inline
bool Argument<std::string>::read( std::list<std::string>& str )
{
    if (str.empty()) return false;
    std::string s = str.front();
    str.pop_front();
    *ptr = s;
    isSet = true;
    return true;
}

/// General case for printing default value
template<class T> inline
void Argument<T>::printValue() const
{
    std::cout << *ptr << " ";
}

/// General case for printing default value
template<> inline
void Argument<std::vector<std::string > >::printValue() const
{
    for (unsigned int i=0; i<ptr->size(); i++)
        std::cout << (*ptr)[i] << " ";
}


//========================================================================

/** Command line parser

This object parses arguments from a command line or from an input stream.
The arguments are described using a pointer, a short name, a long name and a help message. Mandatory arguments are declared using method "parameter", optional arguments are declared using method "option".
Once all arguments declared, operator () does the parsing.
The special option -h or --help displays help on all arguments.
See examples argumentParserLine_test.cpp and argumentParserFile_test.cpp
@see Argument
*/
class SOFA_HELPER_API ArgumentParser
{
    /// String
    typedef std::string string;
    /// Associate a string with a Argument object
    typedef std::map< string, ArgumentBase* > Map;
    /// short name -> Argument object
    std::map< char, ArgumentBase* > shortName;
    /// long name -> Argument object
    Map longName;

    /// Associate name with boolean value (true iff it is set)
    typedef std::map<ArgumentBase*,bool> SetMap;

    /// Set map (bool true iff parameter is set)
    SetMap parameter_set;

    /// Set of commands
    typedef std::vector<ArgumentBase*> ArgVec;
    /// Set of commands
    ArgVec commands;

    /// Set of remaining file
    std::vector<std::string>* files;

    // help stuff
    string globalHelp;    ///< Overall presentation
    char helpShortName;   ///< short name for help
    string helpLongName;  ///< long name for help

public:

    /// Constructor using a global help string
    ArgumentParser( const string& helpstr="", char hlpShrt='h', const string& hlpLng="help" );

    /// Constructor using a global help string and a list of filenames
    ArgumentParser( std::vector<std::string>* files, const string& helpstr="", char hlpShrt='h', const string& hlpLng="help" );

    /// Constructor using a global help string
    ~ArgumentParser();

    /** Declare an optional argument
    \param ptr pointer to the variable
    \param sho short name
    \param lon long name
    \param help
    */
    template<class T> inline
    ArgumentParser& option( T* ptr, char sho, const char* lon, const char* help )
    {
        string sn, ln(lon), h(help); sn += sho;

        if( sho!=0 && shortName.find(sho) != shortName.end() )
        {
            msg_fatal("ArgumentParser") << "name " << sn << " already used !";
            exit(EXIT_FAILURE);
        }

        if( ln.size()>0 && longName.find(ln) != longName.end() )
        {
            msg_fatal("ArgumentParser") << ln << " already used !" ;
            exit(EXIT_FAILURE);
        }

        if( sho!=0 && sho == helpShortName )
        {
            msg_fatal("ArgumentParser") <<sho << " reserved for help !" ;
            exit(EXIT_FAILURE);
        }
        if( ln.size()>0 && lon == helpLongName )
        {
            msg_fatal("ArgumentParser") << "name " << lon << " reserved for help !" ;
            exit(EXIT_FAILURE);
        }

        ArgumentBase* c = new Argument<T>(ptr,sho,ln,h,false);
        shortName[sho] = c;
        longName[lon] = c;
        commands.push_back(c);
        return (*this);
    }

    /** Declare a mandatory argument
    \param ptr pointer to the variable
    \param sho short name
    \param lon long name
    \param help
    */
    template<class T> inline
    ArgumentParser& parameter( T* ptr, char sho, const char* lon, const char* help )
    {
        string sn, ln(lon), h(help); sn += sho;

        if( sho!=0 && shortName.find(sho) != shortName.end() )
        {
            msg_fatal("ArgumentParser") << "name " << sn << " already used !"  ;
            exit(EXIT_FAILURE);
        }

        if( ln.size()>0 && longName.find(ln) != longName.end() )
        {
            msg_fatal("ArgumentParser") << "name " << ln << " already used !" ;
            exit(EXIT_FAILURE);
        }

        if( sho!=0 && sho == helpShortName )
        {
            msg_error("ArgumentParser") << "name " << sho << " reserved for help !" ;
            exit(EXIT_FAILURE);
        }
        if( ln.size()>0 && lon == helpLongName )
        {
            msg_error("ArgumentParser") << "name " << lon << " reserved for help !" ;
            exit(EXIT_FAILURE);
        }

        ArgumentBase* c = new Argument<T>(ptr,sho,ln,h,true);
        shortName[sho] = c;
        longName[lon] = c;
        commands.push_back(c);
        return (*this);
    }

    /** Parse a command line
    \param argc number of arguments + 1, as usual in C
    \param argv arguments
    */
    void operator () ( int argc, char** argv );

    void operator () ( std::list<std::string> str );

};

/** Parse a command line
\param helpstr General help message
\param hs short name for help
\param hl long name for help
This method frees the programmer from explicitly creating an ArgumentParser, which makes the program (hopefully) more readable. Using this method, the ArgumentParser is transparently created, it receives and processes the arguments, then it is destroyed.
*/
inline ArgumentParser parse( const std::string& helpstr="", char hs='h', const std::string& hl="help" )
{
    return ArgumentParser(helpstr,hs,hl);
}

inline ArgumentParser parse( std::vector<std::string>* files, const std::string& helpstr="", char hs='h', const std::string& hl="help" )
{
    return ArgumentParser(files,helpstr,hs,hl);
}

} // namespace helper

} // namespace sofa

#endif
