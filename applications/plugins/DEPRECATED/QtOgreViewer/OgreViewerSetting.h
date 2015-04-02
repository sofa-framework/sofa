/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_CONFIGURATIONSETTING_OGREVIEWER_H
#define SOFA_COMPONENT_CONFIGURATIONSETTING_OGREVIEWER_H

#include <SofaGraphComponent/ViewerSetting.h>

namespace sofa
{

namespace component
{

namespace configurationsetting
{

class OgreViewerSetting: public ViewerSetting
{
public:
    SOFA_CLASS(OgreViewerSetting,core::objectmodel::ConfigurationSetting);
    OgreViewerSetting();

    bool getShadows() const {return shadows.getValue();}
    void setShadows(bool b) {shadows.setValue(b);};


    /*OgreViewerSetting &addCompositor(const std::string &c)
    {
      compositors.beginEdit()->push_back(c);
      compositors.endEdit();
      return *this;
    }*/

    /*    void setCompositors( const helper::vector< std::string > &c){ compositors.setValue(c);}
        const helper::vector< std::string > &getCompositors() const {return compositors.getValue();};
    */

protected:
    Data< bool > shadows;
    // Data< helper::vector< std::string > > compositors;
};

}

}

}
#endif
