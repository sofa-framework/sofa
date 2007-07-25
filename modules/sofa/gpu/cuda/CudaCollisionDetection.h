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
        Mat3x3f rotation;
        Vec3f translation;
        float margin;
        int nbPoints;
        const void* points;
        int gridnx, gridny, gridnz;
        const void* grid;
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
        int index;
        CudaVector<GPUContact> gpucontacts;
        Test() : index(0) {}
        virtual ~Test() {}
        /// Set the pair of elements to test
        virtual void setElems(sofa::core::CollisionElementIterator elem1, sofa::core::CollisionElementIterator elem2)=0;
        /// Fill the info to send to the graphics cars
        virtual void init(GPUTest& test)=0;
        /// Create the list of SOFA contacts from the contacts detected by the GPU
        virtual void fillContacts(DetectionOutputVector& contacts)=0;
    };

    class RigidRigidTest : public Test
    {
    public:
        CudaRigidDistanceGridCollisionElement elem1;
        CudaRigidDistanceGridCollisionElement elem2;
        /// Set the pair of elements to test
        void setElems(sofa::core::CollisionElementIterator elem1, sofa::core::CollisionElementIterator elem2)
        {
            this->elem1 = CudaRigidDistanceGridCollisionElement(elem1);
            this->elem2 = CudaRigidDistanceGridCollisionElement(elem2);
        }
        /// Fill the info to send to the graphics cars
        void init(GPUTest& test);
        /// Create the list of SOFA contacts from the contacts detected by the GPU
        void fillContacts(DetectionOutputVector& contacts);
    };

    class SphereRigidTest : public Test
    {
    public:
        CudaSphere elem1;
        CudaRigidDistanceGridCollisionElement elem2;
        /// Set the pair of elements to test
        void setElems(sofa::core::CollisionElementIterator elem1, sofa::core::CollisionElementIterator elem2)
        {
            this->elem1 = CudaSphere(elem1);
            this->elem2 = CudaRigidDistanceGridCollisionElement(elem2);
        }
        /// Fill the info to send to the graphics cars
        void init(GPUTest& test);
        /// Create the list of SOFA contacts from the contacts detected by the GPU
        void fillContacts(DetectionOutputVector& contacts);
    };

    typedef std::map< std::pair<core::CollisionModel*, core::CollisionModel*>, vector<Test*> > TestMap;

    TestMap tests;

    virtual void beginNarrowPhase();
    virtual void addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair);
    virtual void endNarrowPhase();

};

} // namespace cuda

} // namespace gpu

} // namespace sofa

#endif
