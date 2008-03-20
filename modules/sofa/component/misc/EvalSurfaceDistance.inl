#ifndef SOFA_COMPONENT_MISC_EVALSURFACEDISTANCE_INL
#define SOFA_COMPONENT_MISC_EVALSURFACEDISTANCE_INL

#include "EvalSurfaceDistance.h"
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/helper/gl/template.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

template<class DataTypes>
EvalSurfaceDistance<DataTypes>::EvalSurfaceDistance()
    : maxDist( initData(&maxDist, 1.0, "maxDist", "alarm distance for proximity detection"))
    , surfaceCM(NULL)
    , pointsCM(NULL)
    , intersection(NULL)
    , detection(NULL)
{
}

template<class DataTypes>
EvalSurfaceDistance<DataTypes>::~EvalSurfaceDistance()
{
    if (intersection)
        delete intersection;
    if (detection)
        delete detection;
}

template<class DataTypes>
void EvalSurfaceDistance<DataTypes>::init()
{
    Inherit::init();
    if (!this->mstate1 || !this->mstate2)
        return;
    sofa::core::objectmodel::BaseContext* c1 = this->mstate1->getContext();
    pointsCM = c1->get<sofa::component::collision::PointModel>();
    if (pointsCM == NULL)
    {
        std::cerr << "EvalSurfaceDistance ERROR: object1 PointModel not found."<<std::endl;
        return;
    }
    sofa::core::objectmodel::BaseContext* c2 = this->mstate2->getContext();
    surfaceCM = c2->get<sofa::component::collision::TriangleModel>();
    if (surfaceCM == NULL)
    {
        std::cerr << "EvalSurfaceDistance ERROR: object2 TriangleModel not found."<<std::endl;
        return;
    }

    intersection = new sofa::component::collision::NewProximityIntersection;
    intersection->setContext(getContext());
    intersection->init();

    detection = new sofa::component::collision::BruteForceDetection;
    detection->setContext(getContext());
    detection->init();
}

template<class DataTypes>
double EvalSurfaceDistance<DataTypes>::eval()
{
    if (!this->mstate1 || !this->mstate2 || !surfaceCM || !pointsCM || !intersection || !detection) return 0.0;
    const VecCoord& x1 = *this->mstate1->getX();
    //const VecCoord& x2 = *this->mstate2->getX();
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

    xproj = x1;
    sofa::helper::vector<Real> dmin(xproj.size());
    std::fill(dmin.begin(),dmin.end(),2*maxDist.getValue());

    while (it != itend)
    {
        const ContactVector* contacts = dynamic_cast<const ContactVector*>(it->second);
        if (contacts != NULL)
        {
            std::cout << contacts->size() << " contacts detected." << std::endl;
            for (unsigned int i=0; i<contacts->size(); i++)
            {
                if ((*contacts)[i].elem.first.getCollisionModel() == surfaceCM)
                {
                    if ((*contacts)[i].elem.second.getCollisionModel() == pointsCM)
                    {
                        int pi = (*contacts)[i].elem.second.getIndex();
                        if ((*contacts)[i].value < dmin[pi])
                        {
                            dmin[pi] = (*contacts)[i].value;
                            xproj[pi] = (*contacts)[i].point[0];
                        }
                    }
                }
                else if ((*contacts)[i].elem.second.getCollisionModel() == surfaceCM)
                {
                    if ((*contacts)[i].elem.first.getCollisionModel() == pointsCM)
                    {
                        int pi = (*contacts)[i].elem.first.getIndex();
                        if ((*contacts)[i].value < dmin[pi])
                        {
                            dmin[pi] = (*contacts)[i].value;
                            xproj[pi] = (*contacts)[i].point[1];
                        }
                    }
                }
            }
        }
        it++;
    }
    return this->doEval(x1, xproj);
}

template<class DataTypes>
void EvalSurfaceDistance<DataTypes>::draw()
{
    if (!this->mstate1 || !this->mstate2 || xproj.empty()) return;
    const VecCoord& x1 = *this->mstate1->getX();
    const VecCoord& x2 = xproj; //*this->mstate2->getX();
    this->doDraw(x1, x2);
}

} // namespace misc

} // namespace component

} // namespace sofa

#endif
