/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_SIMULATION_TREE_DDGNODE_H
#define SOFA_SIMULATION_TREE_DDGNODE_H

namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
 *  \brief Abstract base to manage data dependencies. BaseData and DataEngine inherites from this class
 *
 */
class SOFA_CORE_API DDGNode
{
public:

    typedef std::list<DDGNode*> DDGNodeList;

    /// Constructor
    DDGNode():dirty(false) {};

    /// Destructor. Do nothing
    virtual ~DDGNode() {};

    /// Update the value of Datas
    virtual void update() = 0;

    /// True if the Data has been modified
    virtual void setDirty() = 0;

protected:

    void setDirty(DDGNodeList& list)
    {
        if (!dirty)
        {
            dirty = true;
            for(DDGNodeList::iterator it=list.begin(); it!=list.end(); ++it)
            {
                (*it)->setDirty();
            }
        }
    }

    bool dirty;
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
