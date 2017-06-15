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
#include <SofaValidation/DevAngleCollisionMonitor.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace misc
{

template <class DataTypes>
DevAngleCollisionMonitor<DataTypes>::DevAngleCollisionMonitor()
    : maxDist( initData(&maxDist, (Real)1.0, "maxDist", "alarm distance for proximity detection"))
    , pointsCM(NULL)
    , surfaceCM(NULL)
    , intersection(NULL)
    , detection(NULL)
{
}

template <class DataTypes>
void DevAngleCollisionMonitor<DataTypes>::init()
{
    if (!this->mstate1 || !this->mstate2)
    {
        serr << "DevAngleCollisionMonitor ERROR: mstate1 or mstate2 not found."<<sendl;
        return;
    }

    sofa::core::objectmodel::BaseContext* c1 = this->mstate1->getContext();
    c1->get(pointsCM, core::objectmodel::BaseContext::SearchDown);
    if (pointsCM == NULL)
    {
        serr << "DevAngleCollisionMonitor ERROR: object1 PointModel not found."<<sendl;
        return;
    }
    sofa::core::objectmodel::BaseContext* c2 = this->mstate2->getContext();
    c2->get(surfaceCM, core::objectmodel::BaseContext::SearchDown);
    if (surfaceCM == NULL)
    {
        serr << "DevAngleCollisionMonitor ERROR: object2 TriangleModel not found."<<sendl;
        return;
    }

    intersection = sofa::core::objectmodel::New<sofa::component::collision::NewProximityIntersection>();
    this->addSlave(intersection);
    intersection->init();

    detection = sofa::core::objectmodel::New<sofa::component::collision::BruteForceDetection>();
    this->addSlave(detection);
    detection->init();
}

template <class DataTypes>
void DevAngleCollisionMonitor<DataTypes>::eval()
{
    if (!this->mstate1 || !this->mstate2 || !surfaceCM || !pointsCM || !intersection || !detection) return;

    const VecCoord& x = this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    surfaceCM->computeBoundingTree(6);
    pointsCM->computeBoundingTree(6);
    intersection->setAlarmDistance(maxDist.getValue());
    intersection->setContactDistance(0.0);
    detection->setInstance(this);
    detection->setIntersectionMethod(intersection.get());
    sofa::helper::vector<std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> > vectCMPair;
    vectCMPair.push_back(std::make_pair(surfaceCM->getFirst(), pointsCM->getFirst()));

    detection->beginNarrowPhase();
    sout << "narrow phase detection between " <<surfaceCM->getClassName()<< " and " << pointsCM->getClassName() << sendl;
    detection->addCollisionPairs(vectCMPair);
    detection->endNarrowPhase();

    /// gets the pairs Triangle-Line detected in a radius lower than maxDist
    const core::collision::NarrowPhaseDetection::DetectionOutputMap& detectionOutputs = detection->getDetectionOutputs();

    core::collision::NarrowPhaseDetection::DetectionOutputMap::const_iterator it = detectionOutputs.begin();
    core::collision::NarrowPhaseDetection::DetectionOutputMap::const_iterator itend = detectionOutputs.end();

    while (it != itend)
    {

        const ContactVector* contacts = dynamic_cast<const ContactVector*>(it->second);

        if (contacts != NULL)
        {
            core::collision::DetectionOutput c;

            double minNorm = ((*contacts)[0].point[0] - (*contacts)[0].point[1]).norm();

            sout << contacts->size() << " contacts detected." << sendl;
            for (unsigned int i=0; i<contacts->size(); i++)
            {
                if ((*contacts)[i].elem.first.getCollisionModel() == surfaceCM)
                {
                    if ((*contacts)[i].elem.second.getCollisionModel() == pointsCM)
                    {
                        if ((*contacts)[i].elem.second.getIndex() == ((int)x.size()-1))
                        {
                            double norm = ((*contacts)[i].point[0] - (*contacts)[i].point[1]).norm();
                            if (norm < minNorm)
                            {
                                c = (*contacts)[i];
                                minNorm = norm;
                            }
                        }
                        /*			int pi = (*contacts)[i].elem.second.getIndex();
                        			if ((*contacts)[i].value < dmin[pi])
                        			{
                        			    dmin[pi] = (Real)((*contacts)[i].value);
                        			    xproj[pi] = (*contacts)[i].point[0];
                        			}*/
                    }
                }
                else if ((*contacts)[i].elem.second.getCollisionModel() == surfaceCM)
                {
                    if ((*contacts)[i].elem.first.getCollisionModel() == pointsCM)
                    {
                        if ((*contacts)[i].elem.first.getIndex() == ((int)x.size()-1))
                        {
                            double norm = ((*contacts)[i].point[0] - (*contacts)[i].point[1]).norm();

                            if (norm < minNorm)
                            {
                                c = (*contacts)[i];
                                minNorm = norm;
                            }
                        }

// 			int pi = (*contacts)[i].elem.first.getIndex();
// 			if ((*contacts)[i].value < dmin[pi])
// 			{
// 			    dmin[pi] = (Real)((*contacts)[i].value);
// 			    xproj[pi] = (*contacts)[i].point[1];
// 			}
                    }
                }
            }
            if (c.elem.second.getCollisionModel() == surfaceCM)
            {
                if (c.elem.first.getCollisionModel() == pointsCM)
                {
                    sout << "tip point " << c.point[0] << sendl;
                    sout << "nearest skeleton point " << c.point[1] << sendl;
                }
            }
            else
            {
                if (c.elem.first.getCollisionModel() == surfaceCM)
                {
                    if (c.elem.second.getCollisionModel() == pointsCM)
                    {
                        sout << "tip point " << c.point[1] << sendl;
                        sout << "nearest skeleton point " << c.point[0] << sendl;
                    }
                }
            }
        }
        it++;
    }
}

} // namespace misc

} // namespace component

} // namespace sofa
