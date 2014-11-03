/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/Base.h>

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
        fullpath = parentDataFileName->getFullPath();
        if (this->m_owner)
            this->m_owner->sout << "Updated DataFileName " << this->getName() << " with path " << fullpath << this->m_owner->sendl;
    }
    else
    {
        fullpath = m_values[currentAspect()].getValue();
        if (!fullpath.empty())
            helper::system::DataRepository.findFile(fullpath,"",(this->m_owner ? &(this->m_owner->serr) : &std::cerr));
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
                helper::system::DataRepository.findFile(fullpath[i],"",(this->m_owner ? &(this->m_owner->serr) : &std::cerr));
            }
        }
}

} // namespace objectmodel

} // namespace core

} // namespace sofa
