/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GUI_QT_IMAGEQT_H
#define SOFA_GUI_QT_IMAGEQT_H

#include "SofaGUIQt.h"
#include <sofa/helper/io/Image.h>

#include <iostream>

namespace sofa
{

namespace gui
{

namespace qt
{

class SOFA_SOFAGUIQT_API ImageQt : public sofa::helper::io::Image
{
public:
    ImageQt() {}
    ImageQt(const std::string& filename)
    {
        load(filename);
    }
    bool load(std::string filename);
    static bool Init();
};

} //namespace qt

} //namespace gui

} //namespace sofa

#endif
