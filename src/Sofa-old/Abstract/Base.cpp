//
// C++ Implementation: Base
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include "Base.h"
#include "Sofa/Components/Common/Factory.h"
#include <map>

namespace Sofa
{

namespace Abstract
{

Base::Base()
    : name(dataField(&name,std::string(""),"name","object name"))
{

}

Base::~Base()
{}

std::string Base::getName() const
{
    //if( name.getValue().empty() )
    //    return getTypeName();
    return name.getValue();
}

void Base::setName(const std::string& na)
{
    name.setValue(na);
}

const char* Base::getTypeName() const
{
    //return "UNKNOWN";
    // TODO: change the return type to std::string
    // BUG: this is not threadsafe!!!
    static std::map<const Base*, std::string> typenames;
    std::string& str = typenames[this];
    if (str.empty())
    {
        str = Sofa::Components::Common::gettypename(typeid(*this));
    }
    return str.c_str();
}


} // namespace Abstract

} // namespace Sofa
