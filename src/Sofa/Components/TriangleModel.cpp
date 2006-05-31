#include "TriangleModel.h"
#include "TriangleLoader.h"
#include "CubeModel.h"
#include "Triangle.h"
#include "Sofa/Abstract/CollisionElement.h"
#include "Common/ObjectFactory.h"
#include <vector>

namespace Sofa
{
namespace Components
{

SOFA_DECL_CLASS(Triangle)

void create(TriangleModel*& obj, ObjectDescription* arg)
{
    XML::createWithFilename(obj, arg);
    if (obj!=NULL && arg->getAttribute("dx")!=NULL || arg->getAttribute("dy")!=NULL || arg->getAttribute("dz")!=NULL)
        obj->applyTranslation(atof(arg->getAttribute("dx","0.0")),atof(arg->getAttribute("dy","0.0")),atof(arg->getAttribute("dz","0.0")));
}

Creator< ObjectFactory, TriangleModel > TriangleModelClass("Triangle");

class TriangleModel::Loader : public TriangleLoader
{
public:
    TriangleModel* dest;
    Loader(TriangleModel* dest) : dest(dest) { }
    void addVertices (double x, double y, double z)
    {
        int i = dest->getX()->size();
        dest->resize(i+1);
        (*dest->getX())[i] = Vector3(x,y,z);
    }

    void addTriangle (int idp1, int idp2, int idp3)
    {
        Triangle *t = new Triangle(&(dest->getX()->at(idp1)), &(dest->getX()->at(idp2)), &(dest->getX()->at(idp3)),
                &(dest->getV()->at(idp1)), &(dest->getV()->at(idp2)), &(dest->getV()->at(idp3)),
                dest);
        dest->elems.push_back(t);
    }
};

void TriangleModel::applyTranslation(double dx, double dy, double dz)
{
    Vector3 d(dx,dy,dz);
    VecCoord& x = *getX();
    for (unsigned int i = 0; i < x.size(); i++)
        x[i] += d;
}

void TriangleModel::init(const char* file)
{
    this->resize(0);
    elems.clear();
    Loader loader(this);
    loader.load(file);

    /* if (!readOBJ(file))
    {
    	std::cout << "ERROR while loading Triangle model" << std::endl;
    	exit(-1);
    }*/
}

void TriangleModel::draw()
{
    if (!isActive() || !getContext()->getShowCollisionModels()) return;
    //std::cout << "SPHdraw"<<elems.size()<<std::endl;
    glDisable(GL_LIGHTING);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    if (isStatic())
        glColor3f(0.5, 0.5, 0.5);
    else
        glColor3f(1.0, 0.0, 0.0);
    for (unsigned int i=0; i<elems.size(); i++)
    {
        static_cast<Triangle*>(elems[i])->draw();
    }
    if (getPrevious()!=NULL && dynamic_cast<Abstract::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<Abstract::VisualModel*>(getPrevious())->draw();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

/*
void TriangleModel::computeSphereVolume (void)
{
	Vector3 minBB, maxBB;

	std::vector<Vector3*>::iterator it = vertices.begin();
	std::vector<Vector3*>::iterator itEnd = vertices.end();

	if (it != itEnd)
	{
		minBB = *(*it);
		maxBB = *(*it);
		it++;
	}

	for (;it != itEnd; it++)
	{
		minBB.x() = (minBB.x() > (*it)->x()) ? (*it)->x() : minBB.x();
		minBB.y() =  (minBB.y() > (*it)->y()) ? (*it)->y() : minBB.y();
		minBB.z() =  (minBB.z() > (*it)->z()) ? (*it)->y() : minBB.y();

		maxBB.x() =  (maxBB.x() < (*it)->x()) ? (*it)->x() : maxBB.x();
		maxBB.y() =  (maxBB.y() < (*it)->y()) ? (*it)->y() : maxBB.y();
		maxBB.z() =  (maxBB.z() < (*it)->y()) ? (*it)->y() : maxBB.y();
	}

	Vector3 center = Vector3(minBB + maxBB) / 2;
	double radius = (maxBB - center).Length();
	Sphere boundingSphere(center, radius, 0);
	CollisionModel *cm = getFirst();

	SPHmodel *sphModel = NULL;
	if (cm)
	{
		sphModel = cm->getSPHmodel();
		if (sphModel) delete sphModel;
	}

	sphModel = new SPHmodel();
	sphModel->addSphere (boundingSphere);
	previous = sphModel;
	sphModel->setNext(this);
}
*/
void TriangleModel::computeContinueBoundingBox (void)
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

    /* Vector3 minBB, maxBB;
    Vector3 minBBMoved, maxBBMoved;
    std::vector<Vector3> newVertices;
    int size = vertices.size(); */
    std::vector<Vector3> newVertices;

    std::vector<Vector3> &verts = *(this->getX());
    std::vector<Vector3> &velocityVerts = *(this->getV());

    int size = verts.size();

    for (int i = 0; i < size; i++)
    {
        Vector3 newPos = verts[i];
        newPos += velocityVerts[i] * getContext()->getDt();
        newVertices.push_back(newPos);
    }
    Vector3 minBB, maxBB, minBBMoved, maxBBMoved;
    findBoundingBox(verts, minBB, maxBB);
    findBoundingBox(newVertices, minBBMoved, maxBBMoved);

    // get the min max vector with minBB, minBBMoved, maxBB, maxBBMoved
    for (int i = 0; i < 3; i++)
    {
        minBB[i] = (minBB[i] > minBBMoved[i]) ? minBBMoved[i] : minBB[i];
        maxBB[i] = (maxBB[i] > maxBBMoved[i]) ? maxBBMoved[i] : maxBB[i];
    }

    cubeModel->setCube(0,minBB, maxBB);
}


void TriangleModel::computeBoundingBox(void)
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

    Vector3 minBB, maxBB;

    findBoundingBox(*(this->getX()), minBB, maxBB);

    //std::cout << "BBox: <"<<minBB[0]<<','<<minBB[1]<<','<<minBB[2]<<">-<"<<maxBB[0]<<','<<maxBB[1]<<','<<maxBB[2]<<">\n";

    cubeModel->setCube(0,minBB, maxBB);
}

void TriangleModel::beginIntegration(double dt)
{
    //std::cout << "BEGIN"<<std::endl;
    f = internalForces;
    this->Core::MechanicalObject<Vec3Types>::beginIntegration(dt);
}

void TriangleModel::endIntegration(double dt)
{
    this->Core::MechanicalObject<Vec3Types>::endIntegration(dt);
    //std::cout << "END"<<std::endl;
    f = externalForces;
    externalForces->clear();
}

void TriangleModel::accumulateForce()
{
    if (!externalForces->empty())
    {
        //std::cout << "Adding external forces"<<std::endl;
        for (unsigned int i=0; i < externalForces->size(); i++)
            (*getF())[i] += (*externalForces)[i];
    }
    this->Core::MechanicalObject<Vec3Types>::accumulateForce();
}

void TriangleModel::findBoundingBox(const std::vector<Vector3> &verts, Vector3 &minBB, Vector3 &maxBB)
{
    //std::vector<Vector3*>::const_iterator it = points.begin();
    //std::vector<Vector3*>::const_iterator itEnd = points.end();
    //std::vector<Vector3>* verts = this->getX();

    std::vector<Vector3>::const_iterator it = verts.begin();
    std::vector<Vector3>::const_iterator itEnd = verts.end();

    if (it != itEnd)
    {
        minBB = *it;
        maxBB = *it;
        it++;
    }

    for (; it != itEnd; it++)
    {
        minBB.x() =  (minBB.x() > (*it).x()) ? (*it).x() : minBB.x();
        minBB.y() =  (minBB.y() > (*it).y()) ? (*it).y() : minBB.y();
        minBB.z() =  (minBB.z() > (*it).z()) ? (*it).z() : minBB.z();

        maxBB.x() =  (maxBB.x() < (*it).x()) ? (*it).x() : maxBB.x();
        maxBB.y() =  (maxBB.y() < (*it).y()) ? (*it).y() : maxBB.y();
        maxBB.z() =  (maxBB.z() < (*it).z()) ? (*it).z() : maxBB.z();
    }
}

} // namespace Components

} // namespace Sofa

