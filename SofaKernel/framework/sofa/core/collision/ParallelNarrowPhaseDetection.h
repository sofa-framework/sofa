/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_COLLISION_PARALLELNARROWPHASEDETECTION_H
#define SOFA_COMPONENT_COLLISION_PARALLELNARROWPHASEDETECTION_H

#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/ParallelCollisionModel.h>
#include <vector>
#include <set>
#include <map>
#include <algorithm>

namespace sofa
{

namespace core
{

namespace collision
{
#ifdef SOFA_SMP
template<class T>
struct ParallelAddCollisionPair
{
    void operator()(T* d, core::ParallelCollisionModel* cm1, core::ParallelCollisionModel* cm2, a1::Shared_r<bool> cm1Ready, a1::Shared_r<bool> cm2Ready, DetectionOutputVector** outptr)
    {
        cm1Ready.read();
        cm2Ready.read();
        std::pair<core::CollisionModel*, core::CollisionModel* > cmPair(cm1->getFirst(), cm2->getFirst());
        if (*outptr) (*outptr)->clear();
        d->addCollisionPair(cmPair, outptr);
    }
};
#endif
class ParallelNarrowPhaseDetection : virtual public NarrowPhaseDetection
{
public:
    typedef std::map< std::pair<core::CollisionModel*, core::CollisionModel* >, DetectionOutputVector* > DetectionOutputMap;
    typedef std::set< std::pair<core::CollisionModel*, core::CollisionModel* > > PairSet;
protected:
    /// Destructor
    virtual ~ParallelNarrowPhaseDetection() { }
public:
    /// Compute the final output pair at which the result of the collision between the two given model
    /// @return false if no collisions will be computed
    virtual bool getOutputPair(std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair) = 0;

    /// Compute collisions between the given models, storing the pointer to the outputs in outptr
    /// This method should not modify any internal class data, as it can be used in parallel
    virtual void addCollisionPair(const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair, DetectionOutputVector** outptr ) const = 0;

    /// Compute collisions between the given models, storing the pointer to the outputs in outptr
    virtual void addCollisionPair(const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair)
    {
        std::pair<core::CollisionModel*, core::CollisionModel*> cmFinal(cmPair.first->getLast(), cmPair.second->getLast());
        if (!getOutputPair(cmFinal)) return;
        addCollisionPair(cmPair, &(outputsMap[cmFinal]));
    }
#ifdef SOFA_SMP
    virtual void parallelClearOutputs()
    {
        parallelOutputs.clear();
        parallelOutputsTemp.clear();
    }

    virtual void parallelCreateOutputs(core::ParallelCollisionModel* cm1, core::ParallelCollisionModel* cm2, bool keepAlive = true)
    {
        std::pair<core::CollisionModel*, core::CollisionModel*> cmFinal(cm1->getLast(), cm2->getLast());
        if (!getOutputPair(cmFinal)) return;
        if (keepAlive)
            parallelOutputs.insert(cmFinal);
        else
            parallelOutputsTemp.insert(cmFinal);
    }

    virtual void parallelAddCollisionPair(core::ParallelCollisionModel* cm1, core::ParallelCollisionModel* cm2, a1::Shared<bool>& cm1Ready, a1::Shared<bool>& cm2Ready)
    {
        std::pair<core::CollisionModel*, core::CollisionModel*> cmFinal(cm1->getLast(), cm2->getLast());
        if (!getOutputPair(cmFinal)) return;
        parallelCreateOutputs(cm1, cm2);
        core::ParallelCollisionModel* maincm;
        if (!cm1->isSimulated()) maincm = cm2;
        else if (!cm2->isSimulated()) maincm = cm1;
        else if (cm1->getSize() >= cm2->getSize()) maincm = cm1;
        else maincm = cm2;
        maincm->Task< ParallelAddCollisionPair<ParallelNarrowPhaseDetection> >(this, cm1, cm2, cm1Ready, cm2Ready, &(outputsMap[cmFinal]));
    }
#endif
    /// Clear all the potentially colliding pairs detected in the previous simulation step
    virtual void beginNarrowPhase()
    {
        for (DetectionOutputMap::iterator it = outputsMap.begin(); it!=outputsMap.end(); it++)
        {
            if (it->second && !parallelOutputs.count(it->first) && !parallelOutputsTemp.count(it->first))
                it->second->clear();
        }
    }

    /// Add a new list of potentially colliding pairs of models
    virtual void addCollisionPairs(const sofa::helper::vector< std::pair<core::CollisionModel*, core::CollisionModel*> >& v)
    {
        for (sofa::helper::vector< std::pair<core::CollisionModel*, core::CollisionModel*> >::const_iterator it = v.begin(); it!=v.end(); it++)
        {
            std::pair<core::CollisionModel*, core::CollisionModel*> cmFinal(it->first->getLast(), it->second->getLast());
            if (!getOutputPair(cmFinal)) continue;
            if (!parallelOutputs.count(cmFinal) && !parallelOutputsTemp.count(cmFinal))
                addCollisionPair(*it, &(outputsMap[cmFinal]));
        }
    }

    virtual void endNarrowPhase()
    {
        parallelOutputsTemp.clear(); // temporary parallel tasks are recreated at each step
        DetectionOutputMap::iterator it = outputsMap.begin();
        while(it!=outputsMap.end())
        {
            if ((!it->second || it->second->empty()) && !parallelOutputs.count(it->first))
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

protected:
    //DetectionOutputMap outputsMap;
    PairSet parallelOutputs;
    PairSet parallelOutputsTemp;

    //std::map<Instance, DetectionOutputMap> storedOutputsMap;
    std::map<Instance, PairSet> storedParallelOutputs;
    std::map<Instance, PairSet> storedParallelOutputsTemp;

    virtual void changeInstanceNP(Instance inst)
    {
        storedOutputsMap[instance].swap(outputsMap);
        outputsMap.swap(storedOutputsMap[inst]);
        storedParallelOutputs[instance].swap(parallelOutputs);
        parallelOutputs.swap(storedParallelOutputs[inst]);
        storedParallelOutputsTemp[instance].swap(parallelOutputsTemp);
        parallelOutputsTemp.swap(storedParallelOutputsTemp[inst]);
    }
};

} // namespace collision

} // namespace core

} // namespace sofa

#endif
