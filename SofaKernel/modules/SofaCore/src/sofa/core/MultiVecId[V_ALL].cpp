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
#define SOFA_CORE_MULTIVECID_TEMPLATE_DEFINITION
#include <sofa/core/MultiVecId[V_ALL].h>
#include <sofa/core/BaseState.h>

namespace sofa::core
{

template <VecAccess vaccess>
std::string SOFA_CORE_API TMultiVecId<V_ALL, vaccess>::getName() const
{
    if (!hasIdMap())
        return defaultId.getName();
    else
    {
        std::ostringstream out;
        out << '{';
        out << defaultId.getName() << "[*";
        const IdMap& map = getIdMap();
        MyVecId prev = defaultId;
        for (IdMap_const_iterator it = map.begin(), itend = map.end(); it != itend; ++it)
        {
            if (it->second != prev) // new id
            {
                out << "],";
                if (it->second.getType() == defaultId.getType())
                    out << it->second.getIndex();
                else
                    out << it->second.getName();
                out << '[';
                prev = it->second;
            }
            else out << ',';
            if (it->first == nullptr) out << "nullptr";
            else
                out << it->first->getName();
        }
        out << "]}";
        return out.str();
    }
}

template class SOFA_CORE_API TMultiVecId<V_ALL, V_READ>;
template class SOFA_CORE_API TMultiVecId<V_ALL, V_WRITE>;

} // namespace sofa::core

