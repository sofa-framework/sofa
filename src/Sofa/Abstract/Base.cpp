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
    if( name.getValue().empty() )
        return getTypeName();
    return name.getValue();
}

void Base::setName(const std::string& na)
{
    name.setValue(na);
}


} // namespace Abstract

} // namespace Sofa
