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
#include <sofa/helper/system/SetDirectory.h>

#ifndef WIN32
#include <unistd.h>
#else
#include <windows.h>
#include <direct.h>
#endif

#include <string.h>
#include <iostream>

namespace sofa
{

namespace helper
{

namespace system
{

SetDirectory::SetDirectory(const char* filename)
{
    previousDir[0]='\0';
    directory = GetParentDir(filename);
    if (!directory.empty())
    {
        std::cout << "chdir("<<directory<<")"<<std::endl;
#ifndef WIN32
        getcwd(previousDir, sizeof(previousDir));
        chdir(directory.c_str());
#else
        _getcwd(previousDir, sizeof(previousDir));
        _chdir(directory.c_str());
#endif
    }
}

SetDirectory::~SetDirectory()
{
    if (!directory.empty() && previousDir[0])
    {
        std::cout << "chdir("<<previousDir<<")"<<std::endl;
#ifndef WIN32
        chdir(previousDir);
#else
        _chdir(previousDir);
#endif
    }
}

std::string SetDirectory::GetParentDir(const char* filename)
{
    std::string s = filename;
    std::string::size_type pos = s.find_last_of("/\\");
    if (pos == std::string::npos)
        return ""; // no directory
    else
        return s.substr(0,pos);
}

std::string SetDirectory::GetRelativeFile(const char* filename, const char* basename)
{
    std::string base = GetParentDir(basename);
    std::string s = filename;
    // remove any ".."
    while ((s.substr(0,3)=="../" || s.substr(0,3)=="..\\") && !base.empty())
    {
        s = s.substr(3);
        base = GetParentDir(base.c_str());
    }
    if (base.empty())
        return s;
    else
        return base + "/" + s;
}

} // namespace system

} // namespace helper

} // namespace sofa

