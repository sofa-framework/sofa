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
#ifndef SOFA_GPU_CUDA_CUDACOLLISIONDETECTION_H
#define SOFA_GPU_CUDA_CUDACOLLISIONDETECTION_H

#include <sofa/component/collision/BruteForceDetection.h>
#include <sofa/gpu/cuda/CudaDistanceGridCollisionModel.h>
#include <sofa/gpu/cuda/CudaSphereModel.h>


namespace sofa
{

namespace gpu
{

namespace cuda
{

class CudaCollisionDetection : public sofa::component::collision::BruteForceDetection
{
public:
    typedef sofa::component::collision::BruteForceDetection Inherit;
    struct GPUTest
    {
        void* result;
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

    struct GPUContact
    {
        int p1;
        Vec3f p2;
        float distance;
        Vec3f normal;
    };

    CudaVector<GPUTest> gputests;
    CudaVector<int> gpuresults; ///< number of contact detected on each test

    class Test
    {
    public:
        vector< CudaVector<GPUContact>* > results;
        Test() {}
        virtual ~Test()
        {
            for (unsigned int i=0; i<results.size(); ++i)
                if (results[i])
                    delete results[i];
        }
        virtual bool useGPU()=0;
        /// Returns how many tests are required
        virtual int nbTests()=0;
        /// Fill the info to send to the graphics card
        virtual void init(GPUTest* tests)=0;
        /// Create the list of SOFA contacts from the contacts detected by the GPU
        virtual void fillContacts(DetectionOutputVector& contacts, const int* nresults)=0;
    };

    class CPUTest : public Test
    {
    public:
        CPUTest() {}
        bool useGPU() { return false; }
        int nbTests() { return 0; }
        void init(GPUTest* /*tests*/) {}
        void fillContacts(DetectionOutputVector& /*contacts*/, const int* /*nresults*/) {}
    };

    class RigidRigidTest : public Test
    {
    public:
        CudaRigidDistanceGridCollisionModel* model1;
        CudaRigidDistanceGridCollisionModel* model2;
        RigidRigidTest(CudaRigidDistanceGridCollisionModel* model1, CudaRigidDistanceGridCollisionModel* model2);
        bool useGPU() { return true; }
        /// Returns how many tests are required
        virtual int nbTests();
        /// Fill the info to send to the graphics card
        virtual void init(GPUTest* tests);
        /// Create the list of SOFA contacts from the contacts detected by the GPU
        void fillContacts(DetectionOutputVector& contacts, const int* nresults);

    protected:
        void init(GPUTest& test, CudaVector<GPUContact>& gpucontacts, CudaRigidDistanceGridCollisionElement elem1, CudaRigidDistanceGridCollisionElement elem2);
        void fillContacts(DetectionOutputVector& contacts, int nresults, CudaVector<GPUContact>& gpucontacts, CudaRigidDistanceGridCollisionElement e1, CudaRigidDistanceGridCollisionElement e2, bool invert);
    };

    class SphereRigidTest : public Test
    {
    public:
        CudaSphereModel* model1;
        CudaRigidDistanceGridCollisionModel* model2;
        SphereRigidTest(CudaSphereModel* model1, CudaRigidDistanceGridCollisionModel* model2);
        bool useGPU() { return true; }
        /// Returns how many tests are required
        virtual int nbTests();
        /// Fill the info to send to the graphics card
        virtual void init(GPUTest* tests);
        /// Create the list of SOFA contacts from the contacts detected by the GPU
        void fillContacts(DetectionOutputVector& contacts, const int* nresults);
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
