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
#ifndef SOFA_CORE_PARALLELCOLLISIONMODEL_H
#define SOFA_CORE_PARALLELCOLLISIONMODEL_H
#ifdef SOFA_SMP
#include <sofa/defaulttype/SharedTypes.h>
#endif
#include <sofa/core/CollisionModel.h>


namespace sofa
{

namespace core
{

class SOFA_CORE_API ParallelCollisionModel : public CollisionModel
{
public:

    typedef CollisionElementIterator Iterator;
    typedef topology::BaseMeshTopology Topology;
protected:
    /// Constructor
    ParallelCollisionModel()
    {
    }

    /// Destructor
    virtual ~ParallelCollisionModel() { }
public:
    /// Create or update the bounding volume hierarchy.
    /// The created task must write to the provided Shared<bool> and cummulative Shared<int> variables
#ifdef SOFA_SMP
    virtual void computeBoundingTreeParallel(double dt, int maxDepth, a1::Shared<bool>& res1, a1::Shared<int>* res2=NULL) = 0;
#endif

protected:

};

#ifdef SOFA_SMP
template<class T>
struct ParallelComputeBoundingTree
{
    void operator()(core::CollisionModel* cm, double dt, int maxDepth, Shared_r<T> X, Shared_w<bool> res1, Shared_cw<int> res2)
    {
        X.read();
        if (dt == 0.0) cm->computeBoundingTree(maxDepth);
        else cm->computeContinuousBoundingTree(dt, maxDepth);
        res1.write(true);
        res2.access() += 1;
    }
    void operator()(core::CollisionModel* cm, double dt, int maxDepth, Shared_r<T> X, Shared_w<bool> res1)
    {
        X.read();
        if (dt == 0.0) cm->computeBoundingTree(maxDepth);
        else cm->computeContinuousBoundingTree(dt, maxDepth);
        res1.write(true);
    }
};
#endif

} // namespace core

} // namespace sofa

#endif
