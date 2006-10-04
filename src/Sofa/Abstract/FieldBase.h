//
// C++ Interface: FieldBase
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#pragma once

#include <list>
#include <iostream>

namespace Sofa
{

namespace Abstract
{

/**
Abstract base class for all fields, independently of their type.*/
class FieldBase
{
public:
    /** Constructor
    \param l long name
    \param h help
    \param m true iff the argument is mandatory
    */
    FieldBase( const char* h)
        : help(h)
        , isSet(false)
    {}

    /// Base destructor: does nothing.
    virtual ~FieldBase() {}

    /// Read the command line
    virtual bool read( std::string& str ) = 0;

    /// Print the value of the associated variable
    virtual void printValue( std::ostream& ) const =0;

    /// Print the value of the associated variable
    virtual std::string getValueString() const=0;

    /// Help message
    const char* help;

    /// True iff a value has bee read on the command line
    bool isSet;


};

}

}



