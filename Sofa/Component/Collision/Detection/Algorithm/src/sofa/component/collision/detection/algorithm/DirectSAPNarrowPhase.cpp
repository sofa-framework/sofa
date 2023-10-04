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
#include <sofa/component/collision/detection/algorithm/DirectSAPNarrowPhase.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <unordered_map>

namespace sofa::component::collision::detection::algorithm
{
using namespace sofa::component::collision::geometry;

DirectSAPNarrowPhase::DirectSAPNarrowPhase()
        : d_showOnlyInvestigatedBoxes(initData(&d_showOnlyInvestigatedBoxes, true, "showOnlyInvestigatedBoxes", "Show only boxes which will be sent to narrow phase"))
        , d_nbPairs(initData(&d_nbPairs, 0, "nbPairs", "number of pairs of elements sent to narrow phase"))
        , m_currentAxis(0)
        , m_alarmDist(0)
        , m_alarmDist_d2(0)
        , m_sq_alarmDist(0)
{
    d_nbPairs.setReadOnly(true);
}

void DirectSAPNarrowPhase::reset()
{
    m_endPointContainer.clear();
    m_boxes.clear();
    m_isBoxInvestigated.clear();
    m_sortedEndPoints.clear();
    m_addedCollisionModels.clear();
    m_newCollisionModels.clear();
    m_broadPhaseCollisionModels.clear();
}

void DirectSAPNarrowPhase::createBoxesFromCollisionModels()
{
    sofa::type::vector<CubeCollisionModel*> cube_models;
    cube_models.reserve(m_newCollisionModels.size());

    int totalNbElements = 0;
    for (auto* newCM : m_newCollisionModels)
    {
        if (newCM != nullptr)
        {
            totalNbElements += newCM->getSize();
            cube_models.push_back(dynamic_cast<CubeCollisionModel*>(newCM->getPrevious()));
        }
    }

    m_boxes.reserve(m_boxes.size() + totalNbElements);
    int cur_boxID = static_cast<int>(m_boxes.size());

    for (auto* cm : cube_models)
    {
        if (cm != nullptr)
        {
            for (Size j = 0; j < cm->getSize(); ++j)
            {
                m_endPointContainer.emplace_back();
                EndPoint* min = &m_endPointContainer.back();

                m_endPointContainer.emplace_back();
                EndPoint* max = &m_endPointContainer.back();

                min->setBoxID(cur_boxID);
                max->setBoxID(cur_boxID);
                max->setMax();

                m_sortedEndPoints.push_back(min);
                m_sortedEndPoints.push_back(max);

                m_boxes.emplace_back(Cube(cm, j), min, max);
                ++cur_boxID;
            }
        }
    }

    m_isBoxInvestigated.resize(m_boxes.size(), false);
    m_boxData.resize(m_boxes.size());
}

void DirectSAPNarrowPhase::beginNarrowPhase()
{
    NarrowPhaseDetection::beginNarrowPhase();
    m_broadPhaseCollisionModels.clear();
}

void DirectSAPNarrowPhase::addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair)
{
    m_broadPhaseCollisionModels.insert(cmPair.first);
    m_broadPhaseCollisionModels.insert(cmPair.second);
}

void DirectSAPNarrowPhase::endNarrowPhase()
{
    assert(intersectionMethod != nullptr);
    m_alarmDist = getIntersectionMethod()->getAlarmDistance();
    m_sq_alarmDist = m_alarmDist * m_alarmDist;
    m_alarmDist_d2 = m_alarmDist/2.0;

    checkNewCollisionModels();
    updateBoxes();
    cacheData();
    sortEndPoints();
    narrowCollisionDetectionFromSortedEndPoints();

    NarrowPhaseDetection::endNarrowPhase();
}

void DirectSAPNarrowPhase::checkNewCollisionModels()
{
    SCOPED_TIMER_VARNAME(scopeTimer, "Direct SAP check new cm");
    for (auto *cm : m_broadPhaseCollisionModels)
    {
        auto *last = cm->getLast();
        assert(last != nullptr);
        const auto inserstionResult = m_addedCollisionModels.insert(last);
        if (inserstionResult.second) //insertion success
        {
            m_newCollisionModels.emplace_back(last);
        }
    }

    if (!m_newCollisionModels.empty()) //if a new collision model has been introduced in this time step
    {
        createBoxesFromCollisionModels();
        m_newCollisionModels.clear(); //createBoxesFromCollisionModels will be called again iff new collision models are added
    }
}

int DirectSAPNarrowPhase::greatestVarianceAxis() const
{
    type::Vec3 variance;//variances for each axis
    type::Vec3 mean;//means for each axis

    //computing the mean value of end points on each axis
    for (const auto& dsapBox : m_boxes)
    {
        mean += dsapBox.cube.minVect();
        mean += dsapBox.cube.maxVect();
    }

    const auto nbBoxes = m_boxes.size();
    if (nbBoxes > 0)
    {
        mean[0] /= 2. * static_cast<double>(nbBoxes);
        mean[1] /= 2. * static_cast<double>(nbBoxes);
        mean[2] /= 2. * static_cast<double>(nbBoxes);
    }

    //computing the variance of end points on each axis
    for (const auto& dsapBox : m_boxes)
    {
        const type::Vec3 & min = dsapBox.cube.minVect();
        const type::Vec3 & max = dsapBox.cube.maxVect();

        for (unsigned int j = 0 ; j < 3; ++j)
        {
            variance[j] += std::pow(min[j] - mean[j], 2);
            variance[j] += std::pow(max[j] - mean[j], 2);
        }
    }

    if(variance[0] >= variance[1] && variance[0] >= variance[2])
        return 0;
    if(variance[1] >= variance[2])
        return 1;
    return 2;
}

void DirectSAPNarrowPhase::updateBoxes()
{
    SCOPED_TIMER_VARNAME(scopeTimer, "Direct SAP update boxes");
    m_currentAxis = greatestVarianceAxis();
    for (auto& dsapBox : m_boxes)
    {
        dsapBox.update(m_currentAxis, m_alarmDist_d2);
    }

    //used only for drawing
    m_isBoxInvestigated.resize(m_boxes.size(), false);
    std::fill(m_isBoxInvestigated.begin(), m_isBoxInvestigated.end(), false);
}

bool DirectSAPNarrowPhase::isSquaredDistanceLessThan(const DSAPBox &a, const DSAPBox &b, double threshold)
{
    double dist2 = 0.;

    for (int axis = 0; axis < 3; ++axis)
    {
        dist2 += a.squaredDistance(b, axis);
        if (dist2 > threshold)
        {
            return false;
        }
    }

    return true;
}

void DirectSAPNarrowPhase::cacheData()
{
    SCOPED_TIMER_VARNAME(scopeTimer, "Direct SAP cache");

    unsigned int i{ 0 };
    for (const auto& box : m_boxes)
    {
        auto* cubeCollisionModel = box.cube.getCollisionModel();
        auto* lastCollisionModel = cubeCollisionModel->getLast();
        auto* firstCollisionModel = cubeCollisionModel->getFirst();

        auto& data = m_boxData[i++];
        data.lastCollisionModel = lastCollisionModel;
        data.context = lastCollisionModel->getContext();
        data.doesBoxSelfCollide = lastCollisionModel->getSelfCollision();
        data.isBoxSimulated = lastCollisionModel->isSimulated();
        data.collisionElementIterator = box.cube.getExternalChildren().first;
        data.isInBroadPhase = (m_broadPhaseCollisionModels.find(firstCollisionModel) != m_broadPhaseCollisionModels.end() );
    }
}

void DirectSAPNarrowPhase::sortEndPoints()
{
    SCOPED_TIMER_VARNAME(scopeTimer, "Direct SAP sort");
    std::sort(m_sortedEndPoints.begin(), m_sortedEndPoints.end(), CompPEndPoint());
}

void DirectSAPNarrowPhase::narrowCollisionDetectionFromSortedEndPoints()
{
    SCOPED_TIMER_VARNAME(scopeTimer, "Direct SAP intersection");
    int nbInvestigatedPairs{ 0 };

    std::list<int> activeBoxes;//active boxes are the one that we encoutered only their min (end point), so if there are two boxes b0 and b1,
    //if we encounter b1_min as b0_min < b1_min, on the current axis, the two boxes intersect :  b0_min--------------------b0_max
    //                                                                                                      b1_min---------------------b1_max
    //once we encouter b0_max, b0 will not intersect with nothing (trivial), so we delete it from active_boxes.
    //so the rule is : -every time we encounter a box min end point, we check if it is overlapping with other active_boxes and add the owner (a box) of this end point to
    //                  the active boxes.
    //                 -every time we encounter a max end point of a box, we are sure that we encountered min end point of a box because _end_points is sorted,
    //                  so, we delete the owner box, of this max end point from the active boxes

    // Iterators to activeBoxes are stored in a map for a fast access from a box id
    std::unordered_map<int, decltype(activeBoxes)::const_iterator> activeBoxesIt;

    for (const auto* endPoint : m_sortedEndPoints)
    {
        assert(endPoint != nullptr);

        const int boxId0 = endPoint->boxID();
        const BoxData& data0 = m_boxData[boxId0];
        if (!data0.isInBroadPhase)
        {
            continue;
        }

        if (endPoint->max())
        {
            const auto foundIt = activeBoxesIt.find(endPoint->boxID()); // complexity: Constant on average, worst case linear in the size of the container
            if (foundIt != activeBoxesIt.end())
            {
                //erase the box with id endPoint->boxID() from the list of active boxes
                //the iterator is found from a map
                //with std::list, erasing an element does not invalidate the other iterators
                activeBoxes.erase(foundIt->second);// complexity: Constant
            }
        }
        else //we encounter a min possible intersection between it and active_boxes
        {
            const DSAPBox& box0 = m_boxes[boxId0];
            core::CollisionModel *cm0 = data0.lastCollisionModel;
            const auto collisionElement0 = data0.collisionElementIterator;

            for (const int boxId1 : activeBoxes)
            {
                const BoxData& data1 = m_boxData[boxId1];

                if (!isPairFiltered(data0, data1, box0, boxId1))
                {
                    core::CollisionModel *cm1 = data1.lastCollisionModel;

                    bool swapModels = false;
                    core::collision::ElementIntersector* finalintersector = intersectionMethod->findIntersector(cm0, cm1, swapModels);//find the method for the finnest CollisionModels

                    if (!swapModels && cm0->getClass() == cm1->getClass() && cm0 > cm1)//we do that to have only pair (p1,p2) without having (p2,p1)
                        swapModels = true;

                    if (finalintersector != nullptr)
                    {
                        auto collisionElement1 = data1.collisionElementIterator;

                        auto swappableCm0 = cm0;
                        auto swappableCollisionElement0 = collisionElement0;

                        if (swapModels)
                        {
                            std::swap(swappableCm0, cm1);
                            std::swap(swappableCollisionElement0, collisionElement1);
                        }

                        narrowCollisionDetectionForPair(finalintersector, swappableCm0, cm1, swappableCollisionElement0, collisionElement1);

                        //used only for drawing
                        m_isBoxInvestigated[boxId0] = true;
                        m_isBoxInvestigated[boxId1] = true;

                        ++nbInvestigatedPairs;
                    }
                }

            }
            activeBoxes.push_back(boxId0);// complexity: Constant
            auto last = activeBoxes.end();
            --last;//iterator the last element of the list
            activeBoxesIt.insert({boxId0, last});// complexity: Average case: O(1), worst case O(size())
        }
    }

    d_nbPairs.setValue(nbInvestigatedPairs);
    sofa::helper::AdvancedTimer::valSet("Direct SAP pairs", nbInvestigatedPairs);
}

bool DirectSAPNarrowPhase::isPairFiltered(const BoxData &data0, const BoxData &data1, const DSAPBox &box0, int boxId1) const
{
    if (data0.isBoxSimulated || data1.isBoxSimulated) //is any of the object simulated?
    {
        // do the models belong to the same object? Can both object collide?
        if ((data0.context != data1.context) || data0.doesBoxSelfCollide)
        {
            if (isSquaredDistanceLessThan(box0, m_boxes[boxId1], m_sq_alarmDist))
            {
                return false;
            }
        }
    }
    return true;
}

void DirectSAPNarrowPhase::narrowCollisionDetectionForPair(core::collision::ElementIntersector* intersector,
                                                core::CollisionModel *collisionModel0,
                                                core::CollisionModel *collisionModel1,
                                                core::CollisionElementIterator collisionModelIterator0,
                                                core::CollisionElementIterator collisionModelIterator1)
{
    sofa::core::collision::DetectionOutputVector*& outputs = this->getDetectionOutputs(collisionModel0, collisionModel1);
    intersector->beginIntersect(collisionModel0, collisionModel1, outputs);//creates outputs if null
    intersector->intersect(collisionModelIterator0, collisionModelIterator1, outputs);
}

void DirectSAPNarrowPhase::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowDetectionOutputs())
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();

    std::vector<sofa::type::RGBAColor> colors;

    vparams->drawTool()->setPolygonMode(0, true);
    std::vector<sofa::type::Vec3> vertices;

    unsigned int boxId{ 0 };
    for (const auto& dsapBox : m_boxes)
    {
        const bool isBoxInvestigated = m_isBoxInvestigated[boxId++];
        if (d_showOnlyInvestigatedBoxes.getValue() && !isBoxInvestigated) continue;

        const auto& minCorner = dsapBox.cube.minVect();
        const auto& maxCorner = dsapBox.cube.maxVect();

        vertices.emplace_back(minCorner[0], minCorner[1], minCorner[2]);
        vertices.emplace_back(maxCorner[0], minCorner[1], minCorner[2]);

        vertices.emplace_back(minCorner[0], minCorner[1], minCorner[2]);
        vertices.emplace_back(minCorner[0], maxCorner[1], minCorner[2]);

        vertices.emplace_back(minCorner[0], minCorner[1], minCorner[2]);
        vertices.emplace_back(minCorner[0], minCorner[1], maxCorner[2]);

        vertices.emplace_back(minCorner[0], minCorner[1], maxCorner[2]);
        vertices.emplace_back(maxCorner[0], minCorner[1], maxCorner[2]);

        vertices.emplace_back(minCorner[0], maxCorner[1], minCorner[2]);
        vertices.emplace_back(maxCorner[0], maxCorner[1], minCorner[2]);

        vertices.emplace_back(maxCorner[0], minCorner[1], minCorner[2]);
        vertices.emplace_back(maxCorner[0], maxCorner[1], minCorner[2]);

        vertices.emplace_back(minCorner[0], maxCorner[1], minCorner[2]);
        vertices.emplace_back(minCorner[0], maxCorner[1], maxCorner[2]);

        vertices.emplace_back(maxCorner[0], maxCorner[1], minCorner[2]);
        vertices.emplace_back(maxCorner[0], maxCorner[1], maxCorner[2]);

        vertices.emplace_back(maxCorner[0], minCorner[1], minCorner[2]);
        vertices.emplace_back(maxCorner[0], minCorner[1], maxCorner[2]);

        vertices.emplace_back(minCorner[0], maxCorner[1], maxCorner[2]);
        vertices.emplace_back(maxCorner[0], maxCorner[1], maxCorner[2]);

        vertices.emplace_back(maxCorner[0], minCorner[1], maxCorner[2]);
        vertices.emplace_back(maxCorner[0], maxCorner[1], maxCorner[2]);

        vertices.emplace_back(minCorner[0], minCorner[1], maxCorner[2]);
        vertices.emplace_back(minCorner[0], maxCorner[1], maxCorner[2]);

        if (isBoxInvestigated)
        {
            for (unsigned int i = 0; i < 12; ++i)
            {
                colors.emplace_back(1.0, 0.0, 0.0, 1.0);
            }
        }
        else
        {
            for (unsigned int i = 0; i < 12; ++i)
            {
                colors.emplace_back(0.0, 0.0, 1.0, 1.0);
            }
        }
    }

    vparams->drawTool()->drawLines(vertices, 3, colors);
}

inline void DSAPBox::show()const
{
    msg_info("DSAPBox") <<"MIN "<<cube.minVect()<< msgendl
                        <<"MAX "<<cube.maxVect() ;
}

using namespace sofa::defaulttype;
using namespace collision;

int DirectSAPNarrowPhaseClass = core::RegisterObject("Collision detection using sweep and prune")
        .add< DirectSAPNarrowPhase >()
;

}
