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

namespace sofa::core::objectmodel
{

namespace fs = sofa::helper::system;

DataFileName::DataFileName(const std::string& helpMsg, bool isDisplayed, bool isReadOnly): Inherit(helpMsg, isDisplayed, isReadOnly),
    m_pathType(PathType::FILE)
{
}

DataFileName::DataFileName(const std::string& value, const std::string& helpMsg, bool isDisplayed,
                           bool isReadOnly): Inherit(value, helpMsg, isDisplayed, isReadOnly),
                                             m_pathType(PathType::FILE)
{
    updatePath();
}

DataFileName::DataFileName(const BaseData::BaseInitData& init): Inherit(init),
                                                                m_pathType(PathType::FILE)
{
}

DataFileName::DataFileName(const Inherit::InitData& init): Inherit(init),
                                                           m_pathType(PathType::FILE)
{
    updatePath();
}

void DataFileName::setPathType(PathType pathType)
{
    m_pathType = pathType;
}

PathType DataFileName::getPathType() const
{
    return m_pathType;
}

bool DataFileName::read(const std::string& s )
{
    const bool ret = Inherit::read(s);
    if (ret) updatePath();
    return ret;
}

void DataFileName::endEdit()
{
    updatePath();
    Data::notifyEndEdit();
}

const std::string& DataFileName::getRelativePath() const
{
    this->updateIfDirty();
    return m_relativepath ;
}

const std::string& DataFileName::getFullPath() const
{
    this->updateIfDirty();
    return m_fullpath;
}

const std::string& DataFileName::getAbsolutePath() const
{
    this->updateIfDirty();
    return m_fullpath;
}

const std::string& DataFileName::getExtension() const
{
    this->updateIfDirty();
    return m_extension;
}

void DataFileName::doOnUpdate()
{
    updatePath();
}

void DataFileName::updatePath()
{
    const DataFileName* parentDataFileName = dynamic_cast<DataFileName*>(parentData.getTarget());
    if (parentDataFileName)
    {
        const std::string fullpath = parentDataFileName->getFullPath();
        if (getPathType() != PathType::BOTH && getPathType() != parentDataFileName->getPathType())
        {
            msg_error(this->getName()) << "This DataFileName only accepts " << (getPathType() == PathType::FILE ? "directories" : "files");
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
        {
            std::ostringstream tempOss;
            DataRepository.findFile(fullpath, "", &tempOss);
        }

        if (getPathType() != PathType::BOTH && (fs::FileSystem::exists(fullpath) && ((getPathType() == PathType::DIRECTORY) != fs::FileSystem::isDirectory(fullpath))))
        {
            msg_error(this->getName()) << "This DataFileName only accepts " << (getPathType() == PathType::FILE ? "directories" : "files");
        }
        else
        {
            m_fullpath = fullpath;
            // Update the relative path.
            for(const std::string& path : DataRepository.getPaths() )
            {
                if( m_fullpath.find(path) == 0 )
                {
                    m_relativepath = sofa::helper::system::FileRepository::relativeToPath(m_fullpath, path);
                    break;
                }
            }
            if (m_relativepath.empty())
                m_relativepath = m_value.getValue();

            // Compute the file extension if found.
            const std::size_t found = m_relativepath.find_last_of(".");
            if (found != m_relativepath.npos)
                m_extension = m_relativepath.substr(found + 1);
            else
                m_extension = "";
        }
    }
}
} // namespace sofa::core::objectmodel
