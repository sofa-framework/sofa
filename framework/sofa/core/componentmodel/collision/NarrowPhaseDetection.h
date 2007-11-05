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
#ifndef SOFA_COMPONENT_COLLISION_NARROWPHASEDETECTION_H
#define SOFA_COMPONENT_COLLISION_NARROWPHASEDETECTION_H

#include <sofa/core/componentmodel/collision/Detection.h>
#include <vector>
#include <map>
#include <algorithm>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace collision
{

class NarrowPhaseDetection : virtual public Detection
{
public:
    typedef std::map< std::pair<core::CollisionModel*, core::CollisionModel* >, DetectionOutputVector* > DetectionOutputMap;
protected:
    //sofa::helper::vector< std::pair<core::CollisionElementIterator, core::CollisionElementIterator> > elemPairs;
    DetectionOutputMap outputsMap;

public:
    virtual ~NarrowPhaseDetection() { }

    virtual void beginNarrowPhase()
    {
        for (DetectionOutputMap::iterator it = outputsMap.begin(); it!=outputsMap.end(); it++)
        {
            if (it->second)
                it->second->clear();
        }
    }

    virtual void addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair) = 0;

    virtual void addCollisionPairs(const sofa::helper::vector< std::pair<core::CollisionModel*, core::CollisionModel*> >& v)
    {
        for (sofa::helper::vector< std::pair<core::CollisionModel*, core::CollisionModel*> >::const_iterator it = v.begin(); it!=v.end(); it++)
            addCollisionPair(*it);
    }


    virtual void endNarrowPhase()
    {
        DetectionOutputMap::iterator it = outputsMap.begin();
        while(it!=outputsMap.end())
        {
            if (!it->second || it->second->empty())
            {
                DetectionOutputMap::iterator it2 = it;
                ++it2;
                delete it->second;
                outputsMap.erase(it);
                it = it2;
            }
            else
            {
                ++it;
            }
        }
    }

    //sofa::helper::vector<std::pair<core::CollisionElementIterator, core::CollisionElementIterator> >& getCollisionElementPairs() { return elemPairs; }

    DetectionOutputMap& getDetectionOutputs()
    {
        return outputsMap;
    }
};

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
