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
    : f_name(dataField(&f_name,std::string("NoName"),"name","object name"))
{

}

Base::~Base()
{}

const std::string& Base::getName() const
{
    return name;
}

void Base::setName(const std::string& na)
{
    f_name.setValue(na);
}


} // namespace Abstract

} // namespace Sofa
