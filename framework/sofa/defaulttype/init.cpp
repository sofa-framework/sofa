/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "init.h"

#include <sofa/helper/init.h>

#include <iostream>

namespace sofa
{

namespace defaulttype
{

static bool s_initialized = false;
static bool s_cleanedUp = false;

void init()
{
    if (!s_initialized)
    {
        sofa::helper::init();
        s_initialized = true;
    }
}

bool isInitialized()
{
    return s_initialized;
}

void cleanup()
{
    if (!s_cleanedUp)
    {
        sofa::helper::cleanup();
        s_cleanedUp = true;
    }
}

bool isCleanedUp()
{
    return s_cleanedUp;
}

void checkIfInitialized()
{
    if (!isInitialized())
    {
        std::cerr << "Warning: SofaHelper is not initialized (sofa::helper::init() has never been called).  An application should call the init() function of the higher level Sofa library it uses." << std::endl;
    }
}

} // namespace defaulttype

} // namespace sofa
