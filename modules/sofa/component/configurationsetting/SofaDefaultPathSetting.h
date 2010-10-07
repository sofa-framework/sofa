/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_CONFIGURATIONSETTING_SOFADEFAULTPATH_H
#define SOFA_COMPONENT_CONFIGURATIONSETTING_SOFADEFAULTPATH_H

#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/component/component.h>

namespace sofa
{
namespace core
{
namespace objectmodel
{
class BaseObjectDescription;
}
}

namespace component
{

namespace configurationsetting
{

class SOFA_COMPONENT_CONFIGURATIONSETTING_API SofaDefaultPathSetting: public core::objectmodel::ConfigurationSetting
{
public:
    SOFA_CLASS(SofaDefaultPathSetting,core::objectmodel::ConfigurationSetting);
    SofaDefaultPathSetting();

    void setRecordPath(const std::string& f) {recordPath.setValue(f);}
    const std::string &getRecordPath() const {return recordPath.getValue();}

    void setGnuplotPath(const std::string& f) {gnuplotPath.setValue(f);}
    const std::string &getGnuplotPath() const {return gnuplotPath.getValue();}

    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg );

protected:
    Data<std::string> recordPath;
    Data<std::string> gnuplotPath;
    core::objectmodel::DataFileNameVector envPath;
};

}

}

}
#endif
