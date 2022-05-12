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
#include <sofa/helper/TagFactory.h>
#include <sofa/type/vector.h>
#include <mutex>

namespace sofa::helper
{

class TagList
{
protected:
    /// the list of the tag names. the Ids are the indices in the vector
    type::vector<std::string> m_tagsList;

public:
    TagList() = default;
    explicit TagList(type::vector<std::string> tagsList) : m_tagsList(std::move(tagsList)) {}

    std::size_t getID(const std::string& name);
    std::string getName(std::size_t id);
};

std::size_t TagList::getID(const std::string& name)
{
    if (name.empty()) return 0;

    const auto it = std::find(m_tagsList.begin(), m_tagsList.end(), name);
    if (it != m_tagsList.end())
    {
        return std::distance(m_tagsList.begin(), it);
    }

    m_tagsList.push_back(name);
    return m_tagsList.size() - 1;
}

std::string TagList::getName(const std::size_t id)
{
    if( id < m_tagsList.size() )
        return m_tagsList[id];
    return "";
}


std::mutex s_mutex;

std::size_t TagFactory::getID(const std::string& name)
{
    std::lock_guard lock(s_mutex);
    return getTagList()->getID(name);
}

std::string TagFactory::getName(const std::size_t id)
{
    std::lock_guard lock(s_mutex);
    return getTagList()->getName(id);
}

TagList* TagFactory::getTagList()
{
    static TagList tagList { {"0", "Visual"} };
    return &tagList;
}
} // namespace sofa::helper

