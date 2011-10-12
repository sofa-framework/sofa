/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_NARROWPHASEDETECTION_H
#define SOFA_COMPONENT_COLLISION_NARROWPHASEDETECTION_H

#include <sofa/core/collision/Detection.h>
#include <vector>
#include <map>
#include <algorithm>

namespace sofa
{

namespace core
{

namespace collision
{

/**
* @brief Given a set of potentially colliding pairs of models, compute set of contact points
*/

class NarrowPhaseDetection : virtual public Detection
{
public:
    SOFA_ABSTRACT_CLASS(NarrowPhaseDetection, Detection);

    typedef std::map< std::pair<core::CollisionModel*, core::CollisionModel* >, DetectionOutputVector* > DetectionOutputMap;

    /// Destructor
    virtual ~NarrowPhaseDetection() { }

    /// Clear all the potentially colliding pairs detected in the previous simulation step
    virtual void beginNarrowPhase()
    {
        for (DetectionOutputMap::iterator it = outputsMap.begin(); it!=outputsMap.end(); it++)
        {
            if (it->second)
                it->second->clear();
        }
    }

    /// Add a new potentially colliding pairs of models
    virtual void addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair) = 0;

    /// Add a new list of potentially colliding pairs of models
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

protected:
    DetectionOutputMap outputsMap;
    std::map<Instance, DetectionOutputMap> storedOutputsMap;

    virtual void changeInstanceNP(Instance inst)
    {
        storedOutputsMap[instance].swap(outputsMap);
        outputsMap.swap(storedOutputsMap[inst]);
    }
};

} // namespace collision

} // namespace core

} // namespace sofa

#endif
