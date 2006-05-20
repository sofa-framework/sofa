#include "RayModel.h"
#include "CubeModel.h"
#include "Scene.h"
#include "XML/CollisionNode.h"

#include <GL/glut.h>

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(Ray)

using namespace Common;

void create(RayModel*& obj, XML::Node<Abstract::CollisionModel>* arg)
{
    obj = new RayModel;
    if (obj!=NULL && arg->getAttribute("dx")!=NULL || arg->getAttribute("dy")!=NULL || arg->getAttribute("dz")!=NULL)
        obj->applyTranslation(atof(arg->getAttribute("dx","0.0")),atof(arg->getAttribute("dy","0.0")),atof(arg->getAttribute("dz","0.0")));
}

Creator< XML::CollisionNode::Factory, RayModel > RayModelClass("Ray");

RayModel::RayModel()
    : previous(NULL), next(NULL), object(NULL)
{
    internalForces = f;
    externalForces = new VecCoord();
}

Core::BasicMechanicalModel* RayModel::resize(int size)
{
    this->Core::MechanicalObject<Vec3Types>::resize(size);
    unsigned int size0 = elems.size();
    elems.resize(size/2);
    while (size0 < elems.size())
    {
        elems[size0] = new Ray(1, size0, this);
        ++size0;
    }
    return this;
}

void RayModel::addRay(Vector3 origin, Vector3 direction, double length)
{
    int index = elems.size();
    resize(2*(elems.size()+1));
    getRay(index)->origin() = origin;
    getRay(index)->direction() = direction;
    getRay(index)->l() = length;
}

void RayModel::draw()
{
    if (!isActive() || !Scene::getInstance()->getShowCollisionModels()) return;
    //std::cout << "SPHdraw"<<elems.size()<<std::endl;
    glDisable(GL_LIGHTING);
    if (getObject()==NULL)
        glColor3f(0.5, 0.5, 0.5);
    else
        glColor3f(1.0, 0.0, 0.0);
    for (unsigned int i=0; i<elems.size(); i++)
    {
        static_cast<Ray*>(elems[i])->draw();
    }
    if (getPrevious()!=NULL && dynamic_cast<Abstract::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<Abstract::VisualModel*>(getPrevious())->draw();
}

void RayModel::applyTranslation(double dx, double dy, double dz)
{
    Vector3 d(dx,dy,dz);
    for (unsigned int i = 0; i < elems.size(); i++)
        getRay(i)->origin() += d;
}

void RayModel::computeBoundingBox(void)
{
    CubeModel* cubeModel = dynamic_cast<CubeModel*>(getPrevious());

    if (cubeModel == NULL)
    {
        if (getPrevious() != NULL)
        {
            delete getPrevious();
            setPrevious(NULL);
        }
        cubeModel = new CubeModel();
        cubeModel->setObject(getObject());
        this->setPrevious(cubeModel);
        cubeModel->setNext(this);
    }
    else if (getObject()==NULL) return; // No need to recompute BBox if immobile

    Vector3 minRay, maxRay, minBB, maxBB;
    std::vector<Abstract::CollisionElement*>::iterator it = elems.begin();
    std::vector<Abstract::CollisionElement*>::iterator itEnd = elems.end();

    if (it != itEnd)
    {
        static_cast<Ray*>(*it)->getBBox(minRay, maxRay);
        minBB = minRay;
        maxBB = maxRay;
        ++it;
        while (it != itEnd)
        {
            static_cast<Ray*>(*it)->getBBox(minRay, maxRay);
            for (int i=0; i<3; i++)
            {
                if (minBB[i] > minRay[i])
                    minBB[i] = minRay[i];
                if (maxBB[i] < maxRay[i])
                    maxBB[i] = maxRay[i];
            }
            ++it;
        }
    }

    //std::cout << "BBox: <"<<minBB[0]<<','<<minBB[1]<<','<<minBB[2]<<">-<"<<maxBB[0]<<','<<maxBB[1]<<','<<maxBB[2]<<">\n";

    cubeModel->setCube(0,minBB, maxBB);
}

void RayModel::setObject(Abstract::BehaviorModel* obj)
{
    object = obj;
    this->Core::MechanicalObject<Vec3Types>::setObject(obj);
}

void RayModel::beginIteration(double dt)
{
    //std::cout << "BEGIN"<<std::endl;
    f = internalForces;
    this->Core::MechanicalObject<Vec3Types>::beginIteration(dt);
}

void RayModel::endIteration(double dt)
{
    this->Core::MechanicalObject<Vec3Types>::endIteration(dt);
    //std::cout << "END"<<std::endl;
    f = externalForces;
    externalForces->clear();
}

void RayModel::accumulateForce()
{
    if (!externalForces->empty())
    {
        //std::cout << "Adding external forces"<<std::endl;
        for (unsigned int i=0; i < externalForces->size(); i++)
            (*getF())[i] += (*externalForces)[i];
    }
    this->Core::MechanicalObject<Vec3Types>::accumulateForce();
}

} // namespace Components

} // namespace Sofa
