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
#include <sofa/core/objectmodel/DataFileNameVector.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>

using sofa::helper::system::FileSystem;
using sofa::helper::system::DataRepository;

namespace sofa::core::objectmodel
{
DataFileNameVector::DataFileNameVector(const char* helpMsg, bool isDisplayed, bool isReadOnly): Inherit(helpMsg, isDisplayed, isReadOnly),
    m_pathType(PathType::FILE)
{
}

DataFileNameVector::DataFileNameVector(const sofa::type::SVector<std::string>& value,
    const char* helpMsg, bool isDisplayed, bool isReadOnly): Inherit(value, helpMsg, isDisplayed, isReadOnly),
                                                             m_pathType(PathType::FILE)
{
    updatePath();
}

DataFileNameVector::DataFileNameVector(const BaseData::BaseInitData& init): Inherit(init),
    m_pathType(PathType::FILE)
{
}

DataFileNameVector::DataFileNameVector(const Inherit::InitData& init): Inherit(init),
    m_pathType(PathType::FILE)
{
    updatePath();
}

void DataFileNameVector::endEdit()
{
    updatePath();
    Inherit::endEdit();
}

void DataFileNameVector::addPath(const std::string& v, bool clear)
{
    sofa::type::vector<std::string>& val = *beginEdit();
    if(clear) val.clear();
    val.push_back(v);
    endEdit();
}

void DataFileNameVector::setValueAsString(const std::string& v)
{
    sofa::type::SVector<std::string>& val = *beginEdit();
    val.clear();
    std::istringstream ss( v );
    ss >> val;
    endEdit();
}

bool DataFileNameVector::read(const std::string& s)
{
    const bool ret = Inherit::read(s);
    if (ret || m_fullpath.empty()) updatePath();
    return ret;
}

const std::string& DataFileNameVector::getRelativePath(unsigned i)
{
    return getValue()[i];
}

const std::string& DataFileNameVector::getFullPath(unsigned i) const
{
    this->updateIfDirty();
    return m_fullpath[i];
}

const std::string& DataFileNameVector::getAbsolutePath(unsigned i) const
{
    this->updateIfDirty();
    return m_fullpath[i];
}

void DataFileNameVector::doOnUpdate()
{
    this->updatePath();
}

void DataFileNameVector::setPathType(PathType pathType)
{
    m_pathType = pathType;
}

PathType DataFileNameVector::getPathType() const
{
    return m_pathType;
}

void DataFileNameVector::updatePath()
{
    const DataFileNameVector* parentDataFileNameVector = dynamic_cast<DataFileNameVector*>(parentData.getTarget());
    if (parentDataFileNameVector)
    {
        if (getPathType() != PathType::BOTH && getPathType() != parentDataFileNameVector->getPathType())
        {
            msg_error(this->getName()) << "Cannot retrieve DataFileNames from Parent value: this DataFileName only accepts " << (getPathType() == PathType::DIRECTORY ? "directories" : "files");
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
                std::ostringstream tempOss;
                DataRepository.findFile(m_fullpath[i], "", &tempOss);
                if (getPathType() != PathType::BOTH && (FileSystem::exists(m_fullpath[i]) && ((getPathType() == PathType::DIRECTORY) != FileSystem::isDirectory(m_fullpath[i]))))
                {
                    msg_error(this->getName()) << "This DataFileName only accepts " << (getPathType() == PathType::DIRECTORY ? "directories" : "files");
                    m_fullpath[i] = "";
                }

            }
        }
    }
}
}
