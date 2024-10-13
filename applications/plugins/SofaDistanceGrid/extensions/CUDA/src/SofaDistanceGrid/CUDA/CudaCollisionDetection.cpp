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
#include "CudaCollisionDetection.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

using namespace sofa::core::collision;
using namespace sofa::component::collision::geometry;

extern "C"
{
    void CudaCollisionDetection_runTests(unsigned int nbTests, unsigned int maxPoints, const void* tests, void* nresults);
}

int CudaCollisionDetectionClass = core::RegisterObject("GPU-based collision detection using CUDA")
        .add< CudaCollisionDetection >()
        ;

void CudaCollisionDetection::beginNarrowPhase()
{
    Inherit2::beginNarrowPhase();
    for (TestMap::iterator it = tests.begin(), itend = tests.end(); it != itend; ++it)
        if (it->second.index >= 0)
            it->second.index = -1;
        else
            --it->second.index;
}

void CudaCollisionDetection::addCollisionPair( const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair )
{
    core::CollisionModel *finalcm1 = cmPair.first->getLast();
    core::CollisionModel *finalcm2 = cmPair.second->getLast();

    Entry& entry = tests[std::make_pair(finalcm1, finalcm2)];
    if (entry.test == NULL)
    {
        entry.test = createTest(finalcm1, finalcm2);
    }
    entry.index = 0; // active pair
}

void CudaCollisionDetection::endNarrowPhase()
{
    // first clean up old pairs
    for (TestMap::iterator it = tests.begin(), itend = tests.end(); it != itend;)
        if (it->second.index < -100)
        {
            TestMap::iterator it2 = it;
            ++it;
            tests.erase(it2);
        }
        else ++it;

    // compute number of tests
    int ntests = 0;
    for (TestMap::iterator it = tests.begin(), itend = tests.end(); it != itend; ++it)
        if (it->second.index >= 0)
        {
            Test* test = it->second.test;
            if (test->useGPU())
            {
                it->second.index = ntests;
                ntests += test->init();
            }
        }
    if (ntests > 0)
    {
        // resize vectors
        gputests.fastResize(ntests);
        gpuresults.fastResize(ntests);

        // init tests
        for (TestMap::iterator it = tests.begin(), itend = tests.end(); it != itend; ++it)
            if (it->second.index >= 0)
            {
                Test* test = it->second.test;
                if (test->useGPU())
                {
                    test->fillInfo(&gputests[it->second.index]);
                }
            }

        // Launch GPU test
        int maxPoints = 0;
        for (int t=0; t<ntests; t++)
            if (gputests[t].nbPoints > maxPoints)
                maxPoints = gputests[t].nbPoints;
        CudaCollisionDetection_runTests(ntests, maxPoints, gputests.deviceRead(), gpuresults.deviceWrite());

    }

    // compute CPU-side collisions
    for (TestMap::iterator it = tests.begin(), itend = tests.end(); it != itend; ++it)
        if (it->second.index >= 0)
        {
            Test* test = it->second.test;
            if (!test->useGPU())
            {
                core::CollisionModel *cm1 = it->first.first->getFirst(); //->getNext();
                core::CollisionModel *cm2 = it->first.second->getFirst(); //->getNext();
                addCollisionPair( std::make_pair(cm1, cm2) );
            }
        }

    if (ntests > 0)
    {
        // gather gpu results
        const int* results = (const int*)gpuresults.hostRead();
        int total = 0;
        for (int i=0; i<ntests; i++)
            total += gpuresults[i];
        std::cout << "CudaCollisionDetection: GPU tests found "<<total<<" contacts."<<std::endl;

        for (TestMap::iterator it = tests.begin(), itend = tests.end(); it != itend; ++it)
            if (it->second.index >= 0)
            {
                Test* test = it->second.test;
                if (test->useGPU())
                {
                    GPUOutputVector* tresults = &it->second.test->results;
                    this->getDetectionOutputs(it->first.first, it->first.second) = tresults;
                    int newIndex = 0;
                    for (unsigned int t=0; t<tresults->nbTests(); t++)
                    {
                        tresults->wtest(t).curSize = results[it->second.index + t];
                        tresults->wtest(t).newIndex = newIndex;
                        newIndex += tresults->rtest(t).curSize;
                    }
                    //test->fillContacts(this->outputsMap[it->first], results+it->second.index);
                }
            }
    }

    Inherit2::endNarrowPhase();
}


CudaCollisionDetection::Test* CudaCollisionDetection::createTest(core::CollisionModel* model1, core::CollisionModel* model2)
{
    if (CudaRigidDistanceGridCollisionModel* rigid1 = dynamic_cast<CudaRigidDistanceGridCollisionModel*>(model1))
    {
        if (CudaRigidDistanceGridCollisionModel* rigid2 = dynamic_cast<CudaRigidDistanceGridCollisionModel*>(model2))
            return new RigidRigidTest(rigid1, rigid2);
        else if (CudaSphereCollisionModel* sphere2 = dynamic_cast<CudaSphereCollisionModel*>(model2))
            return new SphereRigidTest(sphere2, rigid1);
        else if (CudaPointCollisionModel* point2 = dynamic_cast<CudaPointCollisionModel*>(model2))
            return new PointRigidTest(point2, rigid1);
    }
    else if (CudaSphereCollisionModel* sphere1 = dynamic_cast<CudaSphereCollisionModel*>(model1))
    {
        if (CudaRigidDistanceGridCollisionModel* rigid2 = dynamic_cast<CudaRigidDistanceGridCollisionModel*>(model2))
            return new SphereRigidTest(sphere1, rigid2);
    }
    else if (CudaPointCollisionModel* point1 = dynamic_cast<CudaPointCollisionModel*>(model1))
    {
        if (CudaRigidDistanceGridCollisionModel* rigid2 = dynamic_cast<CudaRigidDistanceGridCollisionModel*>(model2))
            return new PointRigidTest(point1, rigid2);
    }
    std::cout << "CudaCollisionDetection::CPUTest "<<model1->getClassName()<<" - "<<model2->getClassName()<<std::endl;
    return new CPUTest;
}


CudaCollisionDetection::RigidRigidTest::RigidRigidTest( CudaRigidDistanceGridCollisionModel* model1, CudaRigidDistanceGridCollisionModel* model2 )
    : model1(model1), model2(model2)
{
    std::cout << "CudaCollisionDetection::RigidRigidTest "<<model1->getClassName()<<" - "<<model2->getClassName()<<std::endl;
}

/// Returns how many tests are required
int CudaCollisionDetection::RigidRigidTest::init()
{
    results.clear();
    if (!model1->isActive() || !model2->isActive()) return 0;
    bool useP1 = model1->usePoints.getValue();
    bool useP2 = model2->usePoints.getValue();
    if (!useP1 && !useP2) return 0;
    int i0 = model1->getSize();
    for (CudaRigidDistanceGridCollisionElement e1 = CudaRigidDistanceGridCollisionElement(model1->begin()); e1!=model1->end(); ++e1)
        for (CudaRigidDistanceGridCollisionElement e2 = CudaRigidDistanceGridCollisionElement(model2->begin()); e2!=model2->end(); ++e2)
        {
            CudaDistanceGrid* g1 = e1.getGrid();
            CudaDistanceGrid* g2 = e2.getGrid();
            if (g1 && g2)
            {
                if (useP1 && !g1->meshPts.empty()) results.addTest(std::make_pair(e1.getIndex(), i0+e2.getIndex()), g1->meshPts.size());
                if (useP2 && !g2->meshPts.empty()) results.addTest(std::make_pair(i0+e2.getIndex(), e1.getIndex()), g2->meshPts.size());
            }
        }
    return results.nbTests();
}

/// Fill the info to send to the graphics card
void CudaCollisionDetection::RigidRigidTest::fillInfo(GPUTest* tests)
{
    if (results.nbTests()==0) return;
    GPUContact* gresults = (GPUContact*)results.results.deviceWrite();
    GPUContactPoint* gresults1 = (GPUContactPoint*)results.results1.deviceWrite();
    GPUContactPoint* gresults2 = (GPUContactPoint*)results.results2.deviceWrite();
    int i0 = model1->getSize();
    for (unsigned int i=0; i<results.nbTests(); i++)
    {
        const GPUOutputVector::TestEntry& e = results.rtest(i);
        GPUTest& test = tests[i];
        CudaRigidDistanceGridCollisionElement elem1((e.elems.first  < i0)?model1:model2,(e.elems.first  < i0)?e.elems.first :e.elems.first -i0);
        CudaRigidDistanceGridCollisionElement elem2((e.elems.second < i0)?model1:model2,(e.elems.second < i0)?e.elems.second:e.elems.second-i0);
        const CudaVector<Vec3f>& p1 = elem1.getGrid()->meshPts;
        CudaDistanceGrid& g2 = *elem2.getGrid();
        test.nbPoints = p1.size();
        test.result = gresults + e.firstIndex;
        test.result1 = gresults1 + e.firstIndex;
        test.result2 = gresults2 + e.firstIndex;
        test.points = p1.deviceRead();
        test.radius = NULL;
        test.gridnx = g2.getNx();
        test.gridny = g2.getNy();
        test.gridnz = g2.getNz();
        test.gridbbmin = g2.getBBMin();
        test.gridbbmax = g2.getBBMax();
        test.gridp0 = g2.getPMin();
        test.gridinvdp = g2.getInvCellWidth();
        test.grid = g2.getDists().deviceRead();
        test.margin = 0;
        test.rotation = elem2.getRotation().multTranspose(elem1.getRotation());
        test.translation = elem2.getRotation().multTranspose(elem1.getTranslation()-elem2.getTranslation());
    }
}

CudaCollisionDetection::SphereRigidTest::SphereRigidTest(CudaSphereCollisionModel*model1, CudaRigidDistanceGridCollisionModel* model2 )
    : model1(model1), model2(model2)
{
    std::cout << "CudaCollisionDetection::SphereRigidTest "<<model1->getClassName()<<" - "<<model2->getClassName()<<std::endl;
}


/// Returns how many tests are required
int CudaCollisionDetection::SphereRigidTest::init()
{

    results.clear();
    if (!model1->isActive() || !model2->isActive()) return 0;
    const CudaVector<Vec3f>& p1 = model1->getMechanicalState()->read(core::ConstVecCoordId::position())->getValue();
    if (p1.empty()) return 0;

    for (CudaRigidDistanceGridCollisionElement e2 = CudaRigidDistanceGridCollisionElement(model2->begin()); e2!=model2->end(); ++e2)
    {
        CudaDistanceGrid* g2 = e2.getGrid();
        if (g2)
            results.addTest(std::make_pair(0, e2.getIndex()), p1.size());
    }
    return results.nbTests();
}

/// Fill the info to send to the graphics card
void CudaCollisionDetection::SphereRigidTest::fillInfo(GPUTest* tests)
{

    if (results.nbTests()==0) return;
    GPUContact* gresults = (GPUContact*)results.results.deviceWrite();
    GPUContactPoint* gresults1 = (GPUContactPoint*)results.results1.deviceWrite();
    GPUContactPoint* gresults2 = (GPUContactPoint*)results.results2.deviceWrite();
    const CudaVector<Vec3f>& p1 = model1->getMechanicalState()->read(core::ConstVecCoordId::position())->getValue();

    for (unsigned int i=0; i<results.nbTests(); i++)
    {
        const GPUOutputVector::TestEntry& e = results.rtest(i);
        GPUTest& test = tests[i];
        CudaRigidDistanceGridCollisionElement elem2(model2, e.elems.second);
        CudaDistanceGrid* g2 = elem2.getGrid();
        test.nbPoints = p1.size();
        test.result = gresults + e.firstIndex;
        test.result1 = gresults1 + e.firstIndex;
        test.result2 = gresults2 + e.firstIndex;
        test.points = p1.deviceRead();
        test.radius = model1->getR().deviceRead();
        test.grid = g2->getDists().deviceRead();
        test.gridnx = g2->getNx();
        test.gridny = g2->getNy();
        test.gridnz = g2->getNz();
        test.gridbbmin = g2->getBBMin();
        test.gridbbmax = g2->getBBMax();
        test.gridp0 = g2->getPMin();
        test.gridinvdp = g2->getInvCellWidth();
        test.margin = 0; //model1->getRadius(0);
        test.rotation.transpose(Mat3x3f(elem2.getRotation()));
        test.translation = test.rotation*(-elem2.getTranslation());

    }

}

CudaCollisionDetection::PointRigidTest::PointRigidTest( CudaPointCollisionModel* model1, CudaRigidDistanceGridCollisionModel* model2 )
    : model1(model1), model2(model2)
{
    std::cout << "CudaCollisionDetection::PointRigidTest "<<model1->getClassName()<<" - "<<model2->getClassName()<<std::endl;
}

/// Returns how many tests are required
int CudaCollisionDetection::PointRigidTest::init()
{
    results.clear();
    if (!model1->isActive() || !model2->isActive()) return 0;
    for (CudaPoint e1 = CudaPoint(model1->begin()); e1!=model1->end(); ++e1)
        for (CudaRigidDistanceGridCollisionElement e2 = CudaRigidDistanceGridCollisionElement(model2->begin()); e2!=model2->end(); ++e2)
        {
            CudaDistanceGrid* g2 = e2.getGrid();
            if (g2)
                results.addTest(std::make_pair(e1.getIndex(), e2.getIndex()), e1.getSize());
        }
    return results.nbTests();
}

/// Fill the info to send to the graphics card
void CudaCollisionDetection::PointRigidTest::fillInfo(GPUTest* tests)
{
    if (results.nbTests()==0) return;
    GPUContact* gresults = (GPUContact*)results.results.deviceWrite();
    GPUContactPoint* gresults1 = (GPUContactPoint*)results.results1.deviceWrite();
    GPUContactPoint* gresults2 = (GPUContactPoint*)results.results2.deviceWrite();
    for (unsigned int i=0; i<results.nbTests(); i++)
    {
        const GPUOutputVector::TestEntry& e = results.rtest(i);
        GPUTest& test = tests[i];
        CudaPoint elem1(model1, e.elems.first);
        CudaRigidDistanceGridCollisionElement elem2(model2,e.elems.second);
        const CudaVector<Vec3f>& p1 = model1->getMechanicalState()->read(core::ConstVecCoordId::position())->getValue();
        CudaDistanceGrid& g2 = *elem2.getGrid();
        test.nbPoints = elem1.getSize();
        test.result = gresults + e.firstIndex;
        test.result1 = gresults1 + e.firstIndex;
        test.result2 = gresults2 + e.firstIndex;
        test.points = p1.deviceReadAt(elem1.i0());
        test.radius = NULL;
        test.gridnx = g2.getNx();
        test.gridny = g2.getNy();
        test.gridnz = g2.getNz();
        test.gridbbmin = g2.getBBMin();
        test.gridbbmax = g2.getBBMax();
        test.gridp0 = g2.getPMin();
        test.gridinvdp = g2.getInvCellWidth();
        test.grid = g2.getDists().deviceRead();
        test.margin = 0;
        test.rotation.transpose(elem2.getRotation());
        test.translation = elem2.getRotation().multTranspose(-elem2.getTranslation());
    }
}

} // namespace cuda

} // namespace gpu

} // namespace sofa
