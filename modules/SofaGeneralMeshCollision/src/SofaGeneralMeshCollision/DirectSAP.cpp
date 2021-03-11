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
#include <SofaGeneralMeshCollision/DirectSAP.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CapsuleModel.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/Intersection.h>
#include <queue>


namespace sofa::component::collision
{

inline void DSAPBox::update(int axis, double alarmDist)
{
    min->value = (cube.minVect())[axis] - alarmDist;
    max->value = (cube.maxVect())[axis] + alarmDist;
}

double DSAPBox::squaredDistance(const DSAPBox & other) const
{
    double dist2 = 0;

    for (int axis = 0; axis < 3; ++axis)
    {
        dist2 += squaredDistance(other, axis);
    }

    return dist2;
}

inline double DSAPBox::squaredDistance(const DSAPBox & other, int axis) const
{
    const defaulttype::Vector3 & min0 = this->cube.minVect();
    const defaulttype::Vector3 & max0 = this->cube.maxVect();
    const defaulttype::Vector3 & min1 = other.cube.minVect();
    const defaulttype::Vector3 & max1 = other.cube.maxVect();

    if(min0[axis] > max1[axis])
    {
        return std::pow(min0[axis] - max1[axis], 2);
    }

    if(min1[axis] > max0[axis])
    {
        return std::pow(min1[axis] - max0[axis], 2);
    }

    return 0;
}

DirectSAP::DirectSAP()
    : d_draw(initData(&d_draw, false, "draw", "enable/disable display of results"))
    , d_showOnlyInvestigatedBoxes(initData(&d_showOnlyInvestigatedBoxes, true, "showOnlyInvestigatedBoxes", "Show only boxes which will be sent to narrow phase"))
    , d_nbPairs(initData(&d_nbPairs, 0, "nbPairs", "number of pairs of elements sent to narrow phase"))
    , box(initData(&box, "box", "if not empty, objects that do not intersect this bounding-box will be ignored"))
    , m_currentAxis(0)
    , _alarmDist(0)
    , _alarmDist_d2(0)
    , _sq_alarmDist(0)
{
    d_nbPairs.setReadOnly(true);
}

void DirectSAP::init()
{
    reinit();
}


void DirectSAP::reinit()
{
    if (box.getValue()[0][0] >= box.getValue()[1][0])
    {
        boxModel.reset();
    }
    else
    {
        if (!boxModel) boxModel = sofa::core::objectmodel::New<CubeCollisionModel>();
        boxModel->resize(1);
        boxModel->setParentOf(0, box.getValue()[0], box.getValue()[1]);
    }
}

void DirectSAP::reset()
{
    m_endPointContainer.clear();
    _boxes.clear();
    _isBoxInvestigated.clear();
    m_sortedEndPoints.clear();
    collisionModels.clear();
}

inline bool DirectSAP::added(core::CollisionModel *cm) const
{
    assert(cm != nullptr);
    return collisionModels.count(cm->getLast()) >= 1;
}

inline void DirectSAP::add(core::CollisionModel *cm)
{
    assert(cm != nullptr);
    collisionModels.insert(cm->getLast());
    _new_cm.push_back(cm->getLast());
}

void DirectSAP::endBroadPhase()
{
    BroadPhaseDetection::endBroadPhase();

    if (_new_cm.empty())
        return;

    createBoxesFromCollisionModels();
    _new_cm.clear(); //createBoxesFromCollisionModels will be called again iff new collision models are added
}

void DirectSAP::createBoxesFromCollisionModels()
{
    //to gain time, we create at the same time all SAPboxes so as to allocate
    //memory the less times
    sofa::helper::vector<CubeCollisionModel*> cube_models;
    cube_models.reserve(_new_cm.size());

    int totalNbElements = 0;
    for (auto* newCM : _new_cm)
    {
        if (newCM != nullptr)
        {
            totalNbElements += newCM->getSize();
            cube_models.push_back(dynamic_cast<CubeCollisionModel*>(newCM->getPrevious()));
        }
    }

    _boxes.reserve(_boxes.size() + totalNbElements);

    int cur_boxID = static_cast<int>(_boxes.size());

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

                _boxes.emplace_back(Cube(cm, j), min, max);
                ++cur_boxID;
            }
        }
    }

    _isBoxInvestigated.resize(_boxes.size(), false);
}

void DirectSAP::addCollisionModel(core::CollisionModel *cm)
{
    assert(cm != nullptr);
    if(!added(cm))
        add(cm);
}

int DirectSAP::greatestVarianceAxis() const
{
    defaulttype::Vector3 variance;//variances for each axis
    defaulttype::Vector3 mean;//means for each axis

    //computing the mean value of end points on each axis
    for (const auto& dsapBox : _boxes)
    {
        mean += dsapBox.cube.minVect();
        mean += dsapBox.cube.maxVect();
    }

    const auto nbBoxes = _boxes.size();
    if (nbBoxes > 0)
    {
        mean[0] /= 2. * static_cast<double>(nbBoxes);
        mean[1] /= 2. * static_cast<double>(nbBoxes);
        mean[2] /= 2. * static_cast<double>(nbBoxes);
    }

    //computing the variance of end points on each axis
    for (const auto& dsapBox : _boxes)
    {
        const defaulttype::Vector3 & min = dsapBox.cube.minVect();
        const defaulttype::Vector3 & max = dsapBox.cube.maxVect();

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

void DirectSAP::update()
{
    m_currentAxis = greatestVarianceAxis();
    for (auto& dsapBox : _boxes)
    {
        dsapBox.update(m_currentAxis, _alarmDist_d2);
    }

    _isBoxInvestigated.resize(_boxes.size(), false);
    std::fill(_isBoxInvestigated.begin(), _isBoxInvestigated.end(), false);
}

void DirectSAP::beginNarrowPhase()
{
    core::collision::NarrowPhaseDetection::beginNarrowPhase();
    _alarmDist = getIntersectionMethod()->getAlarmDistance();
    _sq_alarmDist = _alarmDist * _alarmDist;
    _alarmDist_d2 = _alarmDist/2.0;
    int nbDetectedPairs{ 0 };

    update();

    sofa::helper::AdvancedTimer::stepBegin("Direct SAP std::sort");
    std::sort(m_sortedEndPoints.begin(),m_sortedEndPoints.end(), CompPEndPoint());
    sofa::helper::AdvancedTimer::stepEnd("Direct SAP std::sort");

    sofa::helper::AdvancedTimer::stepBegin("Direct SAP intersection");

    std::deque<int> active_boxes;//active boxes are the one that we encoutered only their min (end point), so if there are two boxes b0 and b1,
                                 //if we encounter b1_min as b0_min < b1_min, on the current axis, the two boxes intersect :  b0_min--------------------b0_max
                                 //                                                                                                      b1_min---------------------b1_max
                                 //once we encouter b0_max, b0 will not intersect with nothing (trivial), so we delete it from active_boxes.
                                 //so the rule is : -every time we encounter a box min end point, we check if it is overlapping with other active_boxes and add the owner (a box) of this end point to
                                 //                  the active boxes.
                                 //                 -every time we encounter a max end point of a box, we are sure that we encountered min end point of a box because _end_points is sorted,
                                 //                  so, we delete the owner box, of this max end point from the active boxes
    for (auto* endPoint : m_sortedEndPoints)
    {
        assert(endPoint != nullptr);
        if (endPoint->max())
        {
            //erase it from the active_boxes
            assert(std::find(active_boxes.begin(), active_boxes.end(), endPoint->boxID()) != active_boxes.end());
            active_boxes.erase(std::find(active_boxes.begin(),active_boxes.end(), endPoint->boxID()));
        }
        else //we encounter a min possible intersection between it and active_boxes
        {
            const int new_box = endPoint->boxID();
            DSAPBox & box0 = _boxes[new_box];

            for (int activeBoxId : active_boxes)
            {
                DSAPBox & box1 = _boxes[activeBoxId];

                core::CollisionModel *finalcm0 = box0.cube.getCollisionModel()->getLast();//get the finnest CollisionModel which is not a CubeModel
                core::CollisionModel *finalcm1 = box1.cube.getCollisionModel()->getLast();

                const bool isAnySimulated = finalcm0->isSimulated() || finalcm1->isSimulated();
                if (isAnySimulated)
                {
                    const bool isSameObject = finalcm0->getContext() == finalcm1->getContext();
                    if ((!isSameObject || finalcm0->canCollideWith(finalcm1)) &&
                        box0.squaredDistance(box1) <= _sq_alarmDist)
                    {
                        bool swapModels = false;
                        core::collision::ElementIntersector* finalintersector = intersectionMethod->findIntersector(finalcm0, finalcm1, swapModels);//find the method for the finnest CollisionModels

                        assert(box0.cube.getExternalChildren().first.getIndex() == box0.cube.getIndex());
                        assert(box1.cube.getExternalChildren().first.getIndex() == box1.cube.getIndex());

                        if (!swapModels && finalcm0->getClass() == finalcm1->getClass() && finalcm0 > finalcm1)//we do that to have only pair (p1,p2) without having (p2,p1)
                            swapModels = true;

                        if (finalintersector != nullptr)
                        {
                            auto collisionElement0 = box0.cube.getExternalChildren().first;
                            auto collisionElement1 = box1.cube.getExternalChildren().first;

                            if (swapModels)
                            {
                                std::swap(finalcm0, finalcm1);
                                std::swap(collisionElement0, collisionElement1);
                            }

                            sofa::core::collision::DetectionOutputVector*& outputs = this->getDetectionOutputs(finalcm0, finalcm1);
                            finalintersector->beginIntersect(finalcm0, finalcm1, outputs);//creates outputs if null
                            finalintersector->intersect(collisionElement0, collisionElement1, outputs);
                            nbDetectedPairs++;

                            _isBoxInvestigated[activeBoxId] = true;
                            _isBoxInvestigated[new_box] = true;
                        }

                    }
                }
            }
            active_boxes.push_back(new_box);
        }
    }
    d_nbPairs.setValue(nbDetectedPairs);
    sofa::helper::AdvancedTimer::valSet("Direct SAP pairs", nbDetectedPairs);
    sofa::helper::AdvancedTimer::stepEnd("Direct SAP intersection");
}

void DirectSAP::draw(const core::visual::VisualParams* vparams)
{
    if (!d_draw.getValue())
        return;

    vparams->drawTool()->saveLastState();
    vparams->drawTool()->disableLighting();

    std::vector<sofa::helper::types::RGBAColor> colors;

    vparams->drawTool()->setPolygonMode(0, true);
    std::vector<sofa::defaulttype::Vector3> vertices;

    unsigned int boxId{ 0 };
    for (const auto& dsapBox : _boxes)
    {
        const bool isBoxInvestigated = _isBoxInvestigated[boxId++];
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
    vparams->drawTool()->restoreLastState();
}

inline void DSAPBox::show()const
{
    msg_info("DSAPBox") <<"MIN "<<cube.minVect()<< msgendl
                        <<"MAX "<<cube.maxVect() ;
}

using namespace sofa::defaulttype;
using namespace collision;

int DirectSAPClass = core::RegisterObject("Collision detection using sweep and prune")
        .add< DirectSAP >()
        ;

} // namespace sofa::component::collision

