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
#include <fstream>

#include "BaseLoader.h"
#include <sofa/helper/system/FileRepository.h>

namespace sofa
{

namespace core
{

namespace loader
{

bool SOFA_CORE_API canLoad(const char* filename);

BaseLoader::BaseLoader(): m_filename(initData(&m_filename,"filename","Filename of the object"))
{
}

BaseLoader::~BaseLoader()
{
}

void BaseLoader::parse(sofa::core::objectmodel::BaseObjectDescription *arg)
{
    objectmodel::BaseObject::parse(arg);
    if (canLoad())
        load();
    else
        sout << "Doing nothing" << sendl;
}


bool BaseLoader::canLoad()
{
    std::string cmd;

    // -- Check filename field:
    if(m_filename.getValue() == "")
    {
        serr << "Error: MeshLoader: No file name given." << sendl;
        return false;
    }


    // -- Check if file exist:
    const char* filename = m_filename.getFullPath().c_str();
    std::string sfilename (filename);

    if (!sofa::helper::system::DataRepository.findFile(sfilename))
    {
        serr << "Error: MeshLoader: File '" << m_filename << "' not found. " << sendl;
        return false;
    }

    std::ifstream file(filename);

    // -- Check if file is readable:
    if (!file.good())
    {
        serr << "Error: MeshLoader: Cannot read file '" << m_filename << "'." << sendl;
        return false;
    }

    // -- Check first line:
    file >> cmd;
    if (cmd.empty())
    {
        serr << "Error: MeshLoader: Cannot read first line in file '" << m_filename << "'." << sendl;
        file.close();
        return false;
    }

    file.close();
    return true;
}


void BaseLoader::setFilename(std::string f)
{
    m_filename.setValue(f);
}

const std::string& BaseLoader::getFilename()
{
    return m_filename.getValue();
}


void BaseLoader::skipToEOL(FILE* f)
{
    int ch;
    while ((ch = fgetc(f)) != EOF && ch != '\n') ;
}


bool BaseLoader::readLine(char* buf, int size, FILE* f)
{
    buf[0] = '\0';
    if (fgets(buf, size, f) == NULL)
        return false;
    if ((int)strlen(buf)==size-1 && buf[size-1] != '\n')
        skipToEOL(f);
    return true;
}

} /// namespace loader

} /// namespace core

} /// namespace sofa

