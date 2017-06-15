/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_COLLISION_NARROWPHASEDETECTION_H
#define SOFA_COMPONENT_COLLISION_NARROWPHASEDETECTION_H

#include <sofa/core/collision/Detection.h>
#include <sofa/helper/map_ptr_stable_compare.h>
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

    typedef sofa::helper::map_ptr_stable_compare< std::pair< core::CollisionModel*, core::CollisionModel* >, DetectionOutputVector* > DetectionOutputMap;

protected:
    /// Destructor
    virtual ~NarrowPhaseDetection() { }
public:
    /// Clear all the potentially colliding pairs detected in the previous simulation step
    virtual void beginNarrowPhase()
    {
        for (DetectionOutputMap::iterator it = m_outputsMap.begin(); it != m_outputsMap.end(); it++)
        {
            DetectionOutputVector *do_vec = (it->second);

            if (do_vec != 0)
                do_vec->clear();
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
        DetectionOutputMap::iterator it = m_outputsMap.begin();
        
        while (it != m_outputsMap.end())
        {
            DetectionOutputVector *do_vec = (it->second);

            if (!do_vec || do_vec->empty())
            {
                /// @todo Optimization
                DetectionOutputMap::iterator iterase = it;
				++it;
				m_outputsMap.erase(iterase);
                if (do_vec) do_vec->release();
            }
            else
            {
                ++it;
            }
        }
    }

    //sofa::helper::vector<std::pair<core::CollisionElementIterator, core::CollisionElementIterator> >& getCollisionElementPairs() { return elemPairs; }

    const DetectionOutputMap& getDetectionOutputs()
    {
        return m_outputsMap;
    }

    DetectionOutputVector*& getDetectionOutputs(CollisionModel *cm1, CollisionModel *cm2)
    {
        std::pair< CollisionModel*, CollisionModel* > cm_pair = std::make_pair(cm1, cm2);

        DetectionOutputMap::iterator it = m_outputsMap.find(cm_pair);

        if (it == m_outputsMap.end())
        {
            // new contact
            it = m_outputsMap.insert( std::make_pair(cm_pair, static_cast< DetectionOutputVector * >(0)) ).first;
        }

        return it->second;
    }

    //Returns true if the last narrow phase detected no collision, to use after endNarrowPhase.
    inline bool zeroCollision()const{
        return m_outputsMap.empty();
    }

protected:
    bool _zeroCollision;//true if the last narrow phase detected no collision, to use after endNarrowPhase

    virtual void changeInstanceNP(Instance inst)
    {
        m_storedOutputsMap[instance].swap(m_outputsMap);
        m_outputsMap.swap(m_storedOutputsMap[inst]);
    }

private:
    std::map<Instance, DetectionOutputMap> m_storedOutputsMap;

    DetectionOutputMap m_outputsMap;
};

} // namespace collision

} // namespace core

} // namespace sofa

#endif
