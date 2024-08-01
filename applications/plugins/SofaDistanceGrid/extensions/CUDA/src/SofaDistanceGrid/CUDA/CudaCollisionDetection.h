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
#ifndef SOFA_GPU_CUDA_CUDACOLLISIONDETECTION_H
#define SOFA_GPU_CUDA_CUDACOLLISIONDETECTION_H

#include <sofa/core/collision/DetectionOutput.h>
#include <SofaDistanceGrid/CUDA/CudaDistanceGridCollisionModel.h>
#include <SofaCUDA/component/collision/geometry/CudaSphereModel.h>
#include <SofaCUDA/component/collision/geometry/CudaPointModel.h>
#include <sofa/component/collision/detection/algorithm/BruteForceBroadPhase.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/gpu/cuda/GPUDetectionOutputVector.h>


namespace sofa
{

namespace core
{

namespace collision
{

using type::Vec3f;
using type::Mat3x3f;
using sofacuda::GPUDetectionOutputVector;

template<>
class TDetectionOutputVector<sofa::gpu::cuda::CudaRigidDistanceGridCollisionModel,sofa::gpu::cuda::CudaRigidDistanceGridCollisionModel> : public GPUDetectionOutputVector
{
};

template<>
class TDetectionOutputVector<sofa::gpu::cuda::CudaSphereCollisionModel, sofa::gpu::cuda::CudaRigidDistanceGridCollisionModel> : public GPUDetectionOutputVector
{
};

template<>
class TDetectionOutputVector<sofa::gpu::cuda::CudaPointCollisionModel,sofa::gpu::cuda::CudaRigidDistanceGridCollisionModel> : public GPUDetectionOutputVector
{
};

} // namespace collision

} // namespace core

namespace gpu
{

namespace cuda
{



class CudaCollisionDetection
        : public sofa::component::collision::detection::algorithm::BruteForceBroadPhase
        , public sofa::core::collision::NarrowPhaseDetection
{
public:
    SOFA_CLASS2(CudaCollisionDetection, sofa::component::collision::detection::algorithm::BruteForceBroadPhase, sofa::core::collision::NarrowPhaseDetection);
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
        CudaSphereCollisionModel* model1;
        CudaRigidDistanceGridCollisionModel* model2;
        SphereRigidTest(CudaSphereCollisionModel *model1, CudaRigidDistanceGridCollisionModel* model2);
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
        CudaPointCollisionModel* model1;
        CudaRigidDistanceGridCollisionModel* model2;
        PointRigidTest(CudaPointCollisionModel* model1, CudaRigidDistanceGridCollisionModel* model2);
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

    virtual void beginNarrowPhase() override;
    virtual void addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair) override;
    virtual void endNarrowPhase() override;

protected:
    Test* createTest(core::CollisionModel* model1, core::CollisionModel* model2);
};

} // namespace cuda

} // namespace gpu

} // namespace sofa

#endif
