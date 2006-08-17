#include "SphereModel.h"
#include "SphereLoader.h"
#include "CubeModel.h"
#include "Common/ObjectFactory.h"

#include <GL/glut.h>

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(Sphere)

using namespace Common;

void create(SphereModel*& obj, ObjectDescription* arg)
{
    obj = new SphereModel(atof(arg->getAttribute("radius","1,0")));
    if (obj!=NULL)
    {
        if (arg->getAttribute("filename"))
            obj->load(arg->getAttribute("filename"));
        obj->setStatic(atoi(arg->getAttribute("static","0"))!=0);
        if (arg->getAttribute("scale")!=NULL)
            obj->applyScale(atof(arg->getAttribute("scale","1.0")));
        if (arg->getAttribute("dx")!=NULL || arg->getAttribute("dy")!=NULL || arg->getAttribute("dz")!=NULL)
            obj->applyTranslation(atof(arg->getAttribute("dx","0.0")),atof(arg->getAttribute("dy","0.0")),atof(arg->getAttribute("dz","0.0")));
    }
}

Creator< ObjectFactory, SphereModel > SphereModelClass("Sphere");

SphereModel::SphereModel(double radius)
    : defaultRadius(radius), static_(false)
{
}

void SphereModel::resize(int size)
{
    this->Core::MechanicalObject<Vec3Types>::resize(size);
    this->Abstract::CollisionModel::resize(size);
    if ((int)radius.size() < size)
    {
        radius.reserve(size);
        while ((int)radius.size() < size)
            radius.push_back(defaultRadius);
    }
    else
    {
        radius.resize(size);
    }
}

int SphereModel::addSphere(const Vector3& pos, double radius)
{
    int i = size;
    resize(i+1);
    setSphere(i, pos, radius);
    return i;
}

void SphereModel:: setSphere(int i, const Vector3& pos, double r)
{
    if ((unsigned)i >= (unsigned) size) return;
    (*this->getX())[i] = pos;
    radius[i] = r;
}

class SphereModel::Loader : public SphereLoader
{
public:
    SphereModel* dest;
    Loader(SphereModel* dest) : dest(dest) { }
    void addSphere(double x, double y, double z, double r)
    {
        dest->addSphere(Vector3(x,y,z),r);
    }
};

bool SphereModel::load(const char* filename)
{
    this->resize(0);
    Loader loader(this);
    return loader.load(filename);
}

void SphereModel::draw(int index)
{
    Vector3 p = (*getX())[index];
    glPushMatrix();
    glTranslated(p[0], p[1], p[2]);
    glutWireSphere(radius[index], 16, 8);
    glPopMatrix();
}

void SphereModel::draw()
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

void SphereModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    if (isStatic() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        VecCoord& x = *getX();
        for (int i=0; i<size; i++)
        {
            double r = radius[i];
            for (int c=0; c<3; c++)
            {
                minElem[c] = x[i][c] - r;
                maxElem[c] = x[i][c] + r;
            }
            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

void SphereModel::computeContinuousBoundingTree(double dt, int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    if (isStatic() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        VecCoord& x = *getX();
        VecDeriv& v = *getV();
        for (int i=0; i<size; i++)
        {
            double r = radius[i];
            for (int c=0; c<3; c++)
            {
                if (v[i][c] < 0)
                {
                    minElem[c] = x[i][c] + v[i][c]*dt - r;
                    maxElem[c] = x[i][c]           + r;
                }
                else
                {
                    minElem[c] = x[i][c]           - r;
                    maxElem[c] = x[i][c] + v[i][c]*dt + r;
                }
            }
            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

} // namespace Components

} // namespace Sofa
