/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/misc/DevAngleCollisionMonitor.h>

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
        std::cerr << "DevAngleCollisionMonitor ERROR: mstate1 or mstate2 not found."<<std::endl;
        return;
    }

    sofa::core::objectmodel::BaseContext* c1 = this->mstate1->getContext();
    c1->get(pointsCM, BaseContext::SearchDown);
    if (pointsCM == NULL)
    {
        std::cerr << "DevAngleCollisionMonitor ERROR: object1 PointModel not found."<<std::endl;
        return;
    }
    sofa::core::objectmodel::BaseContext* c2 = this->mstate2->getContext();
    c2->get(surfaceCM, BaseContext::SearchDown);
    if (surfaceCM == NULL)
    {
        std::cerr << "DevAngleCollisionMonitor ERROR: object2 TriangleModel not found."<<std::endl;
        return;
    }

    intersection = new sofa::component::collision::NewProximityIntersection;
    intersection->setContext(getContext());
    intersection->init();

    detection = new sofa::component::collision::BruteForceDetection;
    detection->setContext(getContext());
    detection->init();
}

template <class DataTypes>
void DevAngleCollisionMonitor<DataTypes>::eval()
{
    if (!this->mstate1 || !this->mstate2 || !surfaceCM || !pointsCM || !intersection || !detection) return;

    const VecCoord& x = *this->mstate1->getX();
    surfaceCM->computeBoundingTree(6);
    pointsCM->computeBoundingTree(6);
    intersection->setAlarmDistance(maxDist.getValue());
    intersection->setContactDistance(0.0);
    detection->setInstance(this);
    detection->setIntersectionMethod(intersection);
    sofa::helper::vector<std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> > vectCMPair;
    vectCMPair.push_back(std::make_pair(surfaceCM->getFirst(), pointsCM->getFirst()));

    detection->beginNarrowPhase();
    std::cout << "narrow phase detection between " <<surfaceCM->getClassName()<< " and " << pointsCM->getClassName() << std::endl;
    detection->addCollisionPairs(vectCMPair);
    detection->endNarrowPhase();

    /// gets the pairs Triangle-Line detected in a radius lower than maxDist
    const core::componentmodel::collision::NarrowPhaseDetection::DetectionOutputMap& detectionOutputs = detection->getDetectionOutputs();

    core::componentmodel::collision::NarrowPhaseDetection::DetectionOutputMap::const_iterator it = detectionOutputs.begin();
    core::componentmodel::collision::NarrowPhaseDetection::DetectionOutputMap::const_iterator itend = detectionOutputs.end();

    while (it != itend)
    {

        const ContactVector* contacts = dynamic_cast<const ContactVector*>(it->second);

        if (contacts != NULL)
        {
            core::componentmodel::collision::DetectionOutput c;

            double minNorm = ((*contacts)[0].point[0] - (*contacts)[0].point[1]).norm();

            std::cout << contacts->size() << " contacts detected." << std::endl;
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
                    std::cout << "tip point " << c.point[0] << std::endl;
                    std::cout << "nearest skeleton point " << c.point[1] << std::endl;
                }
            }
            else
            {
                if (c.elem.first.getCollisionModel() == surfaceCM)
                {
                    if (c.elem.second.getCollisionModel() == pointsCM)
                    {
                        std::cout << "tip point " << c.point[1] << std::endl;
                        std::cout << "nearest skeleton point " << c.point[0] << std::endl;
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
