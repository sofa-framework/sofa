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
#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/Base.h>

using sofa::helper::system::DataRepository ;

namespace sofa
{

namespace core
{

namespace objectmodel
{

bool DataFileName::read(const std::string& s )
{
    bool ret = Inherit::read(s);
    if (ret) updatePath();
    return ret;
}

void DataFileName::updatePath()
{
    DataFileName* parentDataFileName = NULL;
    if (parentData)
        parentDataFileName = dynamic_cast<DataFileName*>(parentData.get());

    if (parentDataFileName)
    {
        m_fullpath = parentDataFileName->getFullPath();
        if (this->m_owner)
            this->m_owner->sout << "Updated DataFileName " << this->getName() << " with path " << m_fullpath << this->m_owner->sendl;
        m_relativepath = parentDataFileName->getRelativePath() ;
    }
    else
    {
        // Update the fullpath.
        m_fullpath = m_values[currentAspect()].getValue();
        if (!m_fullpath.empty())
            DataRepository.findFile(m_fullpath,"",(this->m_owner ? &(this->m_owner->serr.ostringstream()) : &std::cerr));

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
            m_relativepath = m_values[currentAspect()].getValue();

    }
}

void DataFileNameVector::updatePath()
{
    DataFileNameVector* parentDataFileNameVector = NULL;
    if (parentData)
    {
        parentDataFileNameVector = dynamic_cast<DataFileNameVector*>(parentData.get());
    }
    fullpath = m_values[currentAspect()].getValue();
    if (!fullpath.empty())
        for (unsigned int i=0 ; i<fullpath.size() ; i++)
        {
            if (parentDataFileNameVector)
            {
                fullpath[i] = parentDataFileNameVector->getFullPath(i);
            }
            else
            {
                helper::system::DataRepository.findFile(fullpath[i],"",(this->m_owner ? &(this->m_owner->serr.ostringstream()) : &std::cerr));
            }
        }
}

} // namespace objectmodel

} // namespace core

} // namespace sofa
