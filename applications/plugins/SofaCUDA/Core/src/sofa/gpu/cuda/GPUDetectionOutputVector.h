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
#pragma once
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/gpu/cuda/CudaTypes.h>


namespace sofacuda
{

using sofa::type::Vec3f;

/**
 *  \brief Generic description of a contact point using GPU.
 *
 *  Each contact point is described by :
 *
 *  \item p1: index of the collision element
 *  \item P2: position of the contact point
 *  \item distance: estimated penetration distance
 *  \item normal: contact normal in global space
 */
struct GPUContactPoint
{
    Vec3f p;
    int elem;
};

/**
 *  \brief Generic description of a contact using GPU.
 *
 *  Each contact is described by :
 *
 *  \item distance: estimated penetration distance
 *  \item normal: contact normal in global space
 */
struct GPUContact
{
    Vec3f normal;
    float distance;
};


/**
 *  \brief Abstract description of a set of contact point using GPU.
 */
class GPUDetectionOutputVector : public sofa::core::collision::DetectionOutputVector
{
public:
    ~GPUDetectionOutputVector() override
    {
    }

    sofa::gpu::cuda::CudaVector<GPUContactPoint> results1, results2;
    sofa::gpu::cuda::CudaVector<GPUContact> results;
    struct TestEntry
    {
        int firstIndex; ///< Index of the first result in the results array
        int maxSize; ///< Maximum number of contacts resulting from this test
        int curSize; ///< Current number of detected contacts
        int newIndex; ///< Index of the first result in a new compacted array
        std::pair<int,int> elems; ///< Elements
        TestEntry() : firstIndex(0), maxSize(0), curSize(0), newIndex(0), elems(std::make_pair(0,0)) {}
    };
    sofa::gpu::cuda::CudaVector< TestEntry > tests;

    unsigned int size() const
    {
        if (results.empty()) return 0;
        int s = 0;
        for (unsigned int i=0; i<tests.size(); i++)
            s += tests[i].curSize;
        return s;
    }

    void clear()
    {
        results1.clear();
        results2.clear();
        results.clear();
        tests.clear();
    }

    void release()
    {
        // GPU vectors are stored in other data structures, they should not be deleted by the pipeline
    }

    unsigned int nbTests() { return tests.size(); }

    const TestEntry& rtest(int i) { return tests[i]; }
    TestEntry& wtest(int i) { return tests[i]; }

    int addTest(std::pair<int,int> elems, int maxSize)
    {
        int t = tests.size();
        TestEntry e;
        e.elems = elems;
        e.firstIndex = results.size();
        e.maxSize = maxSize;
        e.curSize = 0;
        e.newIndex = 0;
        results.fastResize(e.firstIndex+maxSize);
        results1.fastResize(e.firstIndex+maxSize);
        results2.fastResize(e.firstIndex+maxSize);
        tests.push_back(e);
        return t;
    }

    const GPUContact* get(int i)
    {
        unsigned int t=0;
        while(t<nbTests() && rtest(t).newIndex > i) ++t;
        if (t<nbTests())
            return &(results[rtest(t).firstIndex + (i-rtest(t).newIndex)]);
        else
            return NULL;
    }

    const GPUContactPoint* getP1(int i)
    {
        unsigned int t=0;
        while(t<nbTests() && rtest(t).newIndex > i) ++t;
        if (t<nbTests())
            return &(results1[rtest(t).firstIndex + (i-rtest(t).newIndex)]);
        else
            return NULL;
    }

    const GPUContactPoint* getP2(int i)
    {
        unsigned int t=0;
        while(t<nbTests() && rtest(t).newIndex > i) ++t;
        if (t<nbTests())
            return &(results2[rtest(t).firstIndex + (i-rtest(t).newIndex)]);
        else
            return nullptr;
    }

    /// Const iterator to iterate the detection pairs
    virtual sofa::type::Vec3 getFirstPosition(unsigned idx) override
    {
        return sofa::type::Vec3(getP1(idx)->p[0],getP1(idx)->p[1],getP1(idx)->p[2]);
    }

    /// Const iterator end to iterate the detection pairs
    virtual sofa::type::Vec3 getSecondPosition(unsigned idx) override
    {
        return sofa::type::Vec3(getP2(idx)->p[0],getP2(idx)->p[1],getP2(idx)->p[2]);
    }

};

}

namespace sofa::core::collision
{

using GPUContactPoint = sofacuda::GPUContactPoint;
using GPUContact = sofacuda::GPUContact;
using GPUDetectionOutputVector = sofacuda::GPUDetectionOutputVector;

}
