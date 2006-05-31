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
    XML::createWithFilename(obj, arg);
    if (obj!=NULL && arg->getAttribute("dx")!=NULL || arg->getAttribute("dy")!=NULL || arg->getAttribute("dz")!=NULL)
        obj->applyTranslation(atof(arg->getAttribute("dx","0.0")),atof(arg->getAttribute("dy","0.0")),atof(arg->getAttribute("dz","0.0")));
    obj->setStatic(!stricmp(arg->getAttribute("static","false"),"true"));
}

Creator< ObjectFactory, SphereModel > SphereModelClass("Sphere");

SphereModel::SphereModel(const char* filename)
    : previous(NULL), next(NULL), static_(false)
{
    internalForces = f;
    externalForces = new VecDeriv();
    init(filename);
}

class SphereModel::Loader : public SphereLoader
{
public:
    SphereModel* dest;
    Loader(SphereModel* dest) : dest(dest) { }
    void addSphere(double x, double y, double z, double r)
    {
        dest->getX()->push_back(Vector3(x,y,z));
        dest->getV()->push_back(Vector3(0,0,0));
        dest->getF()->push_back(Vector3(0,0,0));
        dest->getDx()->push_back(Vector3(0,0,0));
        dest->elems.push_back(new Sphere(r,dest->elems.size(),dest));
        dest->resize(dest->elems.size());
    }
};

void SphereModel::init(const char* filename)
{
    this->resize(0);
    elems.clear();
    Loader loader(this);
    loader.load(filename);
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

void SphereModel::beginIntegration(double dt)
{
    //std::cout << "BEGIN"<<std::endl;
    f = internalForces;
    this->Core::MechanicalObject<Vec3Types>::beginIntegration(dt);
}

void SphereModel::endIntegration(double dt)
{
    this->Core::MechanicalObject<Vec3Types>::endIntegration(dt);
    //std::cout << "END"<<std::endl;
    f = externalForces;
    externalForces->clear();
}

void SphereModel::accumulateForce()
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
