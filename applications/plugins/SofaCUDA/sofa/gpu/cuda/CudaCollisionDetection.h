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
#ifndef SOFA_GPU_CUDA_CUDACOLLISIONDETECTION_H
#define SOFA_GPU_CUDA_CUDACOLLISIONDETECTION_H

#include <sofa/core/collision/DetectionOutput.h>
#include <SofaBaseCollision/BruteForceDetection.h>
#include <sofa/gpu/cuda/CudaDistanceGridCollisionModel.h>
#include <sofa/gpu/cuda/CudaSphereModel.h>
#include <sofa/gpu/cuda/CudaPointModel.h>


namespace sofa
{

namespace core
{

namespace collision
{

using defaulttype::Vec3f;

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

/*
struct GPUDetectionOutput
{
    int p1;
    Vec3f p2;
    //union {
        float distance;
    //    int i2;
    //};
    Vec3f normal;
};
*/

/**
 *  \brief Abstract description of a set of contact point using GPU.
 */
class GPUDetectionOutputVector : public DetectionOutputVector
{
public:
    ~GPUDetectionOutputVector()
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
            return NULL;
    }
};

template<>
class TDetectionOutputVector<sofa::gpu::cuda::CudaRigidDistanceGridCollisionModel,sofa::gpu::cuda::CudaRigidDistanceGridCollisionModel> : public GPUDetectionOutputVector
{
};

template<>
class TDetectionOutputVector<sofa::gpu::cuda::CudaSphereModel,sofa::gpu::cuda::CudaRigidDistanceGridCollisionModel> : public GPUDetectionOutputVector
{
};

template<>
class TDetectionOutputVector<sofa::gpu::cuda::CudaPointModel,sofa::gpu::cuda::CudaRigidDistanceGridCollisionModel> : public GPUDetectionOutputVector
{
};

} // namespace collision

} // namespace core

namespace gpu
{

namespace cuda
{



class CudaCollisionDetection : public sofa::component::collision::BruteForceDetection
{
public:
    SOFA_CLASS(CudaCollisionDetection, sofa::component::collision::BruteForceDetection);
    typedef sofa::component::collision::BruteForceDetection Inherit;
    struct GPUTest
    {
        void* result;
        void* result1;
        void* result2;
        const void* points;
        const void* radius;
        const void* grid;
        Mat3x3f rotation;
        Vec3f translation;
        float margin;
        int nbPoints;
        int gridnx, gridny, gridnz;
        Vec3f gridbbmin, gridbbmax;
        Vec3f gridp0, gridinvdp;
    };

    /*struct GPUContact
    {
        int p1;
        Vec3f p2;
        float distance;
        Vec3f normal;
    };*/
    //typedef sofa::core::collision::GPUDetectionOutput GPUContact;
    typedef sofa::core::collision::GPUDetectionOutputVector GPUOutputVector;

    CudaVector<GPUTest> gputests;
    CudaVector<int> gpuresults; ///< number of contact detected on each test

    typedef sofa::core::collision::DetectionOutputVector DetectionOutputVector;

    class Test
    {
    public:
        GPUOutputVector results;
        Test() {}
        virtual ~Test()
        {
        }
        virtual bool useGPU()=0;
        /// Returns how many tests are required
        virtual int init()=0;
        /// Fill the info to send to the graphics card
        virtual void fillInfo(GPUTest* tests)=0;
        /// Create the list of SOFA contacts from the contacts detected by the GPU
        //virtual void fillContacts(DetectionOutputVector& contacts, const int* nresults)=0;
    };

    class CPUTest : public Test
    {
    public:
        CPUTest() {}
        bool useGPU() { return false; }
        int init() { return 0; }
        void fillInfo(GPUTest* /*tests*/) {}
        //void fillContacts(DetectionOutputVector& /*contacts*/, const int* /*nresults*/) {}
    };

    class RigidRigidTest : public Test
    {
    public:
        CudaRigidDistanceGridCollisionModel* model1;
        CudaRigidDistanceGridCollisionModel* model2;
        RigidRigidTest(CudaRigidDistanceGridCollisionModel* model1, CudaRigidDistanceGridCollisionModel* model2);
        bool useGPU() { return true; }
        /// Returns how many tests are required
        virtual int init();
        /// Fill the info to send to the graphics card
        virtual void fillInfo(GPUTest* tests);
        /// Create the list of SOFA contacts from the contacts detected by the GPU
        //void fillContacts(DetectionOutputVector& contacts, const int* nresults);

    protected:
        //void fillInfo(GPUTest& test, CudaVector<GPUContact>& gpucontacts, CudaRigidDistanceGridCollisionElement elem1, CudaRigidDistanceGridCollisionElement elem2);
        //void fillContacts(DetectionOutputVector& contacts, int nresults, CudaVector<GPUContact>& gpucontacts, CudaRigidDistanceGridCollisionElement e1, CudaRigidDistanceGridCollisionElement e2, bool invert);
    };

    class SphereRigidTest : public Test
    {
    public:
        CudaSphereModel* model1;
        CudaRigidDistanceGridCollisionModel* model2;
        SphereRigidTest(CudaSphereModel* model1, CudaRigidDistanceGridCollisionModel* model2);
        bool useGPU() { return true; }
        /// Returns how many tests are required
        virtual int init();
        /// Fill the info to send to the graphics card
        virtual void fillInfo(GPUTest* tests);
        /// Create the list of SOFA contacts from the contacts detected by the GPU
        //void fillContacts(DetectionOutputVector& contacts, const int* nresults);
    };

    class PointRigidTest : public Test
    {
    public:
        CudaPointModel* model1;
        CudaRigidDistanceGridCollisionModel* model2;
        PointRigidTest(CudaPointModel* model1, CudaRigidDistanceGridCollisionModel* model2);
        bool useGPU() { return true; }
        /// Returns how many tests are required
        virtual int init();
        /// Fill the info to send to the graphics card
        virtual void fillInfo(GPUTest* tests);
        /// Create the list of SOFA contacts from the contacts detected by the GPU
        //void fillContacts(DetectionOutputVector& contacts, const int* nresults);

    protected:
        //void fillInfo(GPUTest& test, CudaVector<GPUContact>& gpucontacts, CudaRigidDistanceGridCollisionElement elem1, CudaRigidDistanceGridCollisionElement elem2);
        //void fillContacts(DetectionOutputVector& contacts, int nresults, CudaVector<GPUContact>& gpucontacts, CudaRigidDistanceGridCollisionElement e1, CudaRigidDistanceGridCollisionElement e2, bool invert);
    };

    struct Entry
    {
        int index; // negative if not active
        Test* test;
        Entry() : index(-1), test(NULL) {}
        ~Entry() { if (test!=NULL) delete test; }
    };

    typedef std::map< std::pair<core::CollisionModel*, core::CollisionModel*>, Entry > TestMap;

    TestMap tests;

    virtual void beginNarrowPhase();
    virtual void addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair);
    virtual void endNarrowPhase();

protected:
    Test* createTest(core::CollisionModel* model1, core::CollisionModel* model2);
};

} // namespace cuda

} // namespace gpu

} // namespace sofa

#endif
