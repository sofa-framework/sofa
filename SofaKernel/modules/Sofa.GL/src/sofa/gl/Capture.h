/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/gl/gl.h>
#include <sofa/helper/io/Image.h>
#include <sofa/gl/config.h>

namespace sofa::gl
{

class SOFA_GL_API Capture
{
protected:
    std::string prefix;
    int counter;

public:

    Capture();

    const std::string& getPrefix() const { return prefix; }
    int getCounter() const { return counter; }

    void setPrefix(const std::string v) { prefix=v; }
    void setCounter(int v=-1) { counter = v; }

    std::string findFilename();
    bool saveScreen(const std::string& filename, int compression_level = -1);

    bool saveScreen(int compression_level = -1);
};

} // namespace sofa::gl
