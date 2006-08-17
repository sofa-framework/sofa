#include "RayModel.h"
#include "CubeModel.h"
#include "Common/ObjectFactory.h"

#include <GL/glut.h>

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(Ray)

using namespace Common;

void create(RayModel*& obj, ObjectDescription* arg)
{
    obj = new RayModel(atof(arg->getAttribute("length","1,0")));
    if (obj!=NULL)
    {
        obj->setStatic(atoi(arg->getAttribute("static","0"))!=0);
        if (arg->getAttribute("scale")!=NULL)
            obj->applyScale(atof(arg->getAttribute("scale","1.0")));
        if (arg->getAttribute("dx")!=NULL || arg->getAttribute("dy")!=NULL || arg->getAttribute("dz")!=NULL)
            obj->applyTranslation(atof(arg->getAttribute("dx","0.0")),atof(arg->getAttribute("dy","0.0")),atof(arg->getAttribute("dz","0.0")));
    }
}

Creator< ObjectFactory, RayModel > RayModelClass("Ray");

RayModel::RayModel(double length)
    : defaultLength(length), static_(false)
{
}

void RayModel::resize(int size)
{
    this->Core::MechanicalObject<Vec3Types>::resize(size);
    this->Abstract::CollisionModel::resize(size/2);
    if ((int)length.size() < size/2)
    {
        length.reserve(size/2);
        while ((int)length.size() < size/2)
            length.push_back(defaultLength);
    }
    else
    {
        length.resize(size/2);
    }
}

int RayModel::addRay(const Vector3& origin, const Vector3& direction, double length)
{
    int i = size;
    resize(2*(i+1));
    Ray r = getRay(i);
    r.origin() = origin;
    r.direction() = direction;
    r.l() = length;
    return i;
}

void RayModel::draw(int index)
{
    Ray r(this, index);
    const Vector3& p1 = r.origin();
    const Vector3 p2 = p1 + r.direction()*r.l();
    glBegin(GL_LINES);
    glVertex3dv(p1);
    glVertex3dv(p2);
    glEnd();
}

void RayModel::draw()
{
    if (!isActive() || !getContext()->getShowCollisionModels()) return;
    glDisable(GL_LIGHTING);
    if (isStatic())
        glColor3f(0.5, 0.5, 0.5);
    else
        glColor3f(1.0, 0.0, 0.0);
    for (int i=0; i<size; i++)
    {
        draw(i);
    }
    if (getPrevious()!=NULL && dynamic_cast<Abstract::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<Abstract::VisualModel*>(getPrevious())->draw();
}

void RayModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    if (isStatic() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        for (int i=0; i<size; i++)
        {
            Ray r(this, i);
            const Vector3& o = r.origin();
            const Vector3& d = r.direction();
            const double l = r.l();
            for (int c=0; c<3; c++)
            {
                if (d[c]<0)
                {
                    minElem[c] = o[c] + d[c]*l;
                    maxElem[c] = o[c];
                }
                else
                {
                    minElem[c] = o[c];
                    maxElem[c] = o[c] + d[c]*l;
                }
            }
            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

void RayModel::applyTranslation(double dx, double dy, double dz)
{
    Vector3 d(dx,dy,dz);
    for (int i = 0; i < getNbRay(); i++)
        getRay(i).origin() += d;
}

} // namespace Components

} // namespace Sofa
