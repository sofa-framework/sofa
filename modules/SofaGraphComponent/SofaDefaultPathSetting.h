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
#ifndef SOFA_COMPONENT_CONFIGURATIONSETTING_SOFADEFAULTPATH_H
#define SOFA_COMPONENT_CONFIGURATIONSETTING_SOFADEFAULTPATH_H
#include "config.h"

#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa
{

namespace component
{

namespace configurationsetting
{

///Class for the configuration of default path for sofa application.
class SOFA_GRAPH_COMPONENT_API SofaDefaultPathSetting: public core::objectmodel::ConfigurationSetting
{
public:
    SOFA_CLASS(SofaDefaultPathSetting,core::objectmodel::ConfigurationSetting); ///< Sofa macro to define typedef.
protected:
    SofaDefaultPathSetting();   ///<Default constructor.
public:
    sofa::core::objectmodel::Data<std::string> recordPath;  ///<Path where will be saved the data of the recorded simulation.
    sofa::core::objectmodel::Data<std::string> gnuplotPath; ///<Path where will be saved the gnuplot files.
};

}

}

}
#endif
