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
// Fran�is Faure, iMAGIS-GRAVIR, May 2001
//========================================================

#ifndef SOFAPML_ARGUMENT_PARSER_H
#define SOFAPML_ARGUMENT_PARSER_H

#include <iostream>
#include <stdlib.h>
//#include <strstream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <list>

typedef std::istringstream istrstream;

/// Abstract base class for all command line arguments
struct ArgumentBase
{
    /// character string
    typedef std::string string;

    /** Constructor
    \param s short name
    \param l long name
    \param h help
    \param m true iff the argument is mandatory
    */
    ArgumentBase(char s, string l, string h, bool m)
        : shortName(s)
        , longName(l)
        , help(h)
        , mandatory(m)
        , isSet(false)
    {}

    /// Base destructor: does nothing.
    virtual ~ArgumentBase() {}

    /// Read the command line
    virtual bool read( std::list<std::string>& str ) = 0;

    /// Print the value of the associated variable
    virtual void printValue() const =0;

    char shortName; ///< Short name
    string longName;  ///< Long name
    string help;      ///< Help message

    /// print short name, long name, help
    inline void print () const
    {
        std::cout << "-" << shortName <<",\t--"<< longName <<":\t" << help;
        if( mandatory ) std::cout<< " (required) ";
        std::cout << "  (default: "; printValue();
        std::cout << ")" << std::endl;
    }

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
struct Argument : public ArgumentBase
{


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


//========================================================================

/** Command line parser

This object parses arguments from a command line or from an input stream.
The arguments are described using a pointer, a short name, a long name and a help message. Mandatory arguments are declared using method "parameter", optional arguments are declared using method "option".
Once all arguments declared, operator () does the parsing.
The special option -h or --help displays help on all arguments.
See examples argumentParserLine_test.cpp and argumentParserFile_test.cpp
@see Argument
*/
class ArgumentParser
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

    // help stuff
    string globalHelp;    ///< Overall presentation
    char helpShortName;   ///< short name for help
    string helpLongName;  ///< long name for help

public:

    /// Constructor using a global help string
    ArgumentParser( const string& helpstr="", char hlpShrt='h', const string& hlpLng="help" )
        : globalHelp( helpstr )
        , helpShortName(hlpShrt)
        , helpLongName(hlpLng)
    {}

    /** Declare an optional argument
    \param ptr pointer to the variable
    \param sho short name
    \param lon long name
    \param help
    */
    template<class T> inline
    ArgumentParser& option( T* ptr, char sho, char* lon, char* help )
    {
        string sn, ln(lon), h(help); sn += sho;

        if( sho!=0 && shortName.find(sho) != shortName.end() )
        {
            std::cerr << "name " << sn << " already used !" << std::endl;
            exit( 1 );
        }

        if( ln.size()>0 && longName.find(ln) != longName.end() )
        {
            std::cerr << "name " << ln << " already used !" << std::endl;
            exit( 1 );
        }

        if( sho!=0 && sho == helpShortName )
        {
            std::cerr << "name " << sho << " reserved for help !" << std::endl;
            exit( 1 );
        }
        if( ln.size()>0 && lon == helpLongName )
        {
            std::cerr << "name " << lon << " reserved for help !" << std::endl;
            exit( 1 );
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
    ArgumentParser& parameter( T* ptr, char sho, char* lon, char* help )
    {
        string sn, ln(lon), h(help); sn += sho;

        if( sho!=0 && shortName.find(sho) != shortName.end() )
        {
            std::cerr << "name " << sn << " already used !" << std::endl;
            exit( 1 );
        }

        if( ln.size()>0 && longName.find(ln) != longName.end() )
        {
            std::cerr << "name " << ln << " already used !" << std::endl;
            exit( 1 );
        }

        if( sho!=0 && sho == helpShortName )
        {
            std::cerr << "name " << sho << " reserved for help !" << std::endl;
            exit( 1 );
        }
        if( ln.size()>0 && lon == helpLongName )
        {
            std::cerr << "name " << lon << " reserved for help !" << std::endl;
            exit( 1 );
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
    inline void operator () ( int argc, char** argv )
    {
        std::list<std::string> str;
        for (int i=1; i<argc; ++i)
            str.push_back(std::string(argv[i]));
        (*this)(str);
    }

    inline void operator () ( std::list<std::string> str )
    {
        string shHelp("-");  shHelp.push_back( helpShortName );
        string lgHelp("--"); lgHelp.append( helpLongName );
        string name;
        while( !str.empty() )
        {
            name = str.front();
            str.pop_front();
//			std::cout << "name = " << name << std::endl;
//			std::cout << "lgHelp = " << lgHelp << std::endl;
//			std::cout << "shHelp = " << shHelp << std::endl;

            // display help
            if( name == shHelp || name == lgHelp )
            {
                if( globalHelp.size()>0 ) std::cout<< globalHelp <<std::endl;
                std::cout << "(short name, long name, description, default value)\n-h,\t--help: this help" << std::endl;
                for( ArgVec::const_iterator a=commands.begin(), aend=commands.end(); a!=aend; ++a )
                    (*a)->print();
                exit(1);
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
                        std::cerr<< "\ncould not read value for option " << name << std::endl << std::endl;
                    else parameter_set[longName[ a ]] = true;
                }
                else std::cerr << "\nUnknown option " << name << std::endl << std::endl;
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
                            std::cerr<< "\ncould not read value for option " << name << std::endl << std::endl;
                        else parameter_set[shortName[ a ]] = true;
                    }
                    else std::cerr << "\nUnknown option " << name[i] << std::endl << std::endl;
                }
            }

            //
            else std::cerr << "Unknown option " << name << std::endl;
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
        if( unset ) exit(1);
    }

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




#endif

