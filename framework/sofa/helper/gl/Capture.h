/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_HELPER_GL_CAPTURE_H
#define SOFA_HELPER_GL_CAPTURE_H

#include <sofa/helper/system/gl.h>

#include <sofa/helper/io/Image.h>

namespace sofa
{

namespace helper
{

namespace gl
{

//using namespace sofa::defaulttype;

class Capture
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

} // namespace gl

} // namespace helper

} // namespace sofa

#endif
