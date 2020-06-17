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
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/Base.h>

using sofa::helper::system::DataRepository ;

namespace sofa
{

namespace core
{

namespace objectmodel
{

namespace fs = sofa::helper::system;

DataFileNameVector::~DataFileNameVector()
{
}


bool DataFileName::read(const std::string& s )
{
    bool ret = Inherit::read(s);
    if (ret) updatePath();
    return ret;
}

void DataFileName::updatePath()
{
    DataFileName* parentDataFileName = nullptr;
    if (parentData)
        parentDataFileName = dynamic_cast<DataFileName*>(parentData.get());

    if (parentDataFileName)
    {
        std::string fullpath = parentDataFileName->getFullPath();
        if (isDirectory() != parentDataFileName->isDirectory())
        {
            msg_error(this->getName()) << "This DataFileName only accepts " << (isDirectory() ? "directories" : "files");
        }
        else
        {
            m_fullpath = fullpath;
            m_relativepath = parentDataFileName->getRelativePath();
            m_extension = parentDataFileName->getExtension();
        }
    }
    else
    {
        // Update the fullpath.
        std::string fullpath = m_value.getValue();
        if (!fullpath.empty())
            DataRepository.findFile(fullpath,"",(this->m_owner ? &(this->m_owner->serr.ostringstream()) : &std::cerr));

        if (isDirectory() != fs::FileSystem::isDirectory(fullpath) && fs::FileSystem::exists(fullpath))
        {
            msg_error(this->getName()) << "This DataFileName only accepts " << (isDirectory() ? "directories" : "files");
        }
        else
        {
            m_fullpath = fullpath;
            // Update the relative path.
            for(const std::string& path : DataRepository.getPaths() )
            {
                if( m_fullpath.find(path) == 0 )
                {
                    m_relativepath=DataRepository.relativeToPath(m_fullpath, path,
                                                                 false /*option for backward compatibility*/);
                    break;
                }
            }
            if (m_relativepath.empty())
                m_relativepath = m_value.getValue();

            // Compute the file extension if found.
            std::size_t found = m_relativepath.find_last_of(".");
            if (found != m_relativepath.npos)
                m_extension = m_relativepath.substr(found + 1);
            else
                m_extension = "";
        }
    }
}

void DataFileNameVector::updatePath()
{
    DataFileNameVector* parentDataFileNameVector = nullptr;
    if (parentData)
    {
        parentDataFileNameVector = dynamic_cast<DataFileNameVector*>(parentData.get());
        if (isDirectory() != parentDataFileNameVector->isDirectory())
        {
            msg_error(this->getName()) << "Cannot retrieve DataFileNames from Parent value: this DataFileName only accepts " << (isDirectory() ? "directories" : "files");
            return;
        }
    }
    m_fullpath = m_value.getValue();
    if (!m_fullpath.empty())
    {
        for (unsigned int i=0 ; i<m_fullpath.size() ; i++)
        {
            if (parentDataFileNameVector)
            {
                m_fullpath[i] = parentDataFileNameVector->getFullPath(i);
            }
            else
            {
                helper::system::DataRepository.findFile(m_fullpath[i],"",(this->m_owner ? &(this->m_owner->serr.ostringstream()) : &std::cerr));
                if (isDirectory() != fs::FileSystem::isDirectory(m_fullpath[i]) && fs::FileSystem::exists(m_fullpath[i]))
                {
                    msg_error(this->getName()) << "This DataFileName only accepts " << (isDirectory() ? "directories" : "files");
                    m_fullpath[i] = "";
                }

            }
        }
    }
}

} // namespace objectmodel

} // namespace core

} // namespace sofa
