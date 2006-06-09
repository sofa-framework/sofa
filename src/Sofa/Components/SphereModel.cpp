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
    : previous(NULL), next(NULL), defaultRadius(radius), static_(false)
{
}

void SphereModel::resize(int size)
{
    this->MechanicalObject<Vec3Types>::resize(size);
    int s = this->elems.size();
    if (s < size)
    {
        elems.reserve(size);
        while (s < size)
            elems.push_back(new Sphere(defaultRadius, s++, this));
    }
    else
    {
        while (s > size)
            delete elems[s--];
        elems.resize(size);
    }
}

class SphereModel::Loader : public SphereLoader
{
public:
    SphereModel* dest;
    Loader(SphereModel* dest) : dest(dest) { }
    void addSphere(double x, double y, double z, double r)
    {
        int i = dest->getX()->size();
        dest->resize(i+1);
        (*dest->getX())[i] = Vector3(x,y,z);
        static_cast<Sphere*>(dest->elems[i])->r() = r;
    }
};

bool SphereModel::load(const char* filename)
{
    this->resize(0);
    Loader loader(this);
    return loader.load(filename);
}

void SphereModel::draw()
{
    if (!isActive() || !getContext()->getShowCollisionModels()) return;
    //std::cout << "SPHdraw"<<elems.size()<<std::endl;
    glDisable(GL_LIGHTING);
    if (isStatic())
        glColor3f(0.5, 0.5, 0.5);
    else
        glColor3f(1.0, 0.0, 0.0);
    for (unsigned int i=0; i<elems.size(); i++)
    {
        static_cast<Sphere*>(elems[i])->draw();
    }
    if (getPrevious()!=NULL && dynamic_cast<Abstract::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<Abstract::VisualModel*>(getPrevious())->draw();
}

void SphereModel::applyTranslation(double dx, double dy, double dz)
{
    Vector3 d(dx,dy,dz);
    VecCoord& x = *getX();
    for (unsigned int i = 0; i < x.size(); i++)
        x[i] += d;
}

void SphereModel::applyScale(double s)
{
    VecCoord& x = *getX();
    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i] *= s;
        static_cast<Sphere*>(elems[i])->radius *=s;
    }
}

void SphereModel::computeBoundingBox(void)
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
        cubeModel->setContext(getContext());
        this->setPrevious(cubeModel);
        cubeModel->setNext(this);
    }
    else if (isStatic()) return; // No need to recompute BBox if immobile

    Vector3 minSph, maxSph, minBB, maxBB;
    std::vector<Abstract::CollisionElement*>::iterator it = elems.begin();
    std::vector<Abstract::CollisionElement*>::iterator itEnd = elems.end();

    if (it != itEnd)
    {
        static_cast<Sphere*>(*it)->getBBox(minSph, maxSph);
        minBB = minSph;
        maxBB = maxSph;
        ++it;
        while (it != itEnd)
        {
            static_cast<Sphere*>(*it)->getBBox(minSph, maxSph);
            for (int i=0; i<3; i++)
            {
                if (minBB[i] > minSph[i])
                    minBB[i] = minSph[i];
                if (maxBB[i] < maxSph[i])
                    maxBB[i] = maxSph[i];
            }
            ++it;
        }
    }

    //std::cout << "BBox: <"<<minBB[0]<<','<<minBB[1]<<','<<minBB[2]<<">-<"<<maxBB[0]<<','<<maxBB[1]<<','<<maxBB[2]<<">\n";

    cubeModel->setCube(0,minBB, maxBB);
}

} // namespace Components

} // namespace Sofa
