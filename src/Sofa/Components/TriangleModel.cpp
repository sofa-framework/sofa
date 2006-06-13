#include "TriangleModel.h"
#include "CubeModel.h"
#include "Triangle.h"
#include "Sofa/Abstract/CollisionElement.h"
#include "Common/ObjectFactory.h"
#include <vector>
#include <GL/gl.h>

namespace Sofa
{
namespace Components
{

SOFA_DECL_CLASS(Triangle)

void create(TriangleModel*& obj, ObjectDescription* arg)
{
    obj = new TriangleModel;
    if (obj!=NULL)
    {
        obj->setStatic(atoi(arg->getAttribute("static","0"))!=0);
    }
}

Creator< ObjectFactory, TriangleModel > TriangleModelClass("Triangle");

TriangleModel::TriangleModel()
{
    mmodel = NULL;
    mesh = NULL;
    previous = NULL;
    next = NULL;
    static_ = false;
}

TriangleModel::~TriangleModel()
{
}

void TriangleModel::init()
{
    mmodel = dynamic_cast< Core::MechanicalModel<Vec3Types>* > (getContext()->getMechanicalModel());
    mesh = dynamic_cast< MeshTopology* > (getContext()->getTopology());

    if (mmodel==NULL)
    {
        std::cerr << "ERROR: TriangleModel requires a Vec3 Mechanical Model.\n";
        return;
    }

    if (mesh==NULL)
    {
        std::cerr << "ERROR: TriangleModel requires a Mesh Topology.\n";
        return;
    }
    elems.clear();
    const int npoints = mmodel->getX()->size();
    const int ntris = mesh->getNbTriangles();
    const int nquads = mesh->getNbQuads();
    elems.reserve(ntris+2*nquads);
    //VecCoord& x = *mmodel->getX();
    //VecDeriv& v = *mmodel->getV();
    for (int i=0; i<ntris; i++)
    {
        MeshTopology::Triangle idx = mesh->getTriangle(i);
        if (idx[0] >= npoints || idx[1] >= npoints || idx[2] >= npoints)
        {
            std::cerr << "ERROR: Out of range index in triangle "<<i<<": "<<idx[0]<<" "<<idx[1]<<" "<<idx[2]<<" ( total points="<<npoints<<")\n";
            continue;
        }
        Triangle *t = new Triangle(i, idx[0], idx[1], idx[2], this);
        elems.push_back(t);
    }
    for (int i=0; i<nquads; i++)
    {
        MeshTopology::Quad idx = mesh->getQuad(i);
        if (idx[0] >= npoints || idx[1] >= npoints || idx[2] >= npoints || idx[3] >= npoints)
        {
            std::cerr << "ERROR: Out of range index in quad "<<i<<": "<<idx[0]<<" "<<idx[1]<<" "<<idx[2]<<" "<<idx[3]<<" ( total points="<<npoints<<")\n";
            continue;
        }
        Triangle *t1 = new Triangle(i+ntris, idx[0], idx[1], idx[2], this);
        Triangle *t2 = new Triangle(i+ntris, idx[0], idx[2], idx[3], this);
        elems.push_back(t1);
        elems.push_back(t2);
    }
}

/*
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
	VecCoord& x = *mmodel->getX();
	for (unsigned int i = 0; i < x.size(); i++)
		x[i] += d;
}

void TriangleModel::init(const char* file)
{
	this->resize(0);
	elems.clear();
	Loader loader(this);
	loader.load(file);
}
*/

void TriangleModel::draw()
{
    if (!isActive() || !getContext()->getShowCollisionModels()) return;
    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glEnable(GL_LIGHTING);
    //Enable<GL_BLEND> blending;
    //glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

    static const float color[3] = { 1.0f, 0.0f, 0.0f};
    static const float colorStatic[3] = { 0.5f, 0.5f, 0.5f};
    if (isStatic())
        glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, colorStatic);
    else
        glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);

    for (unsigned int i=0; i<elems.size(); i++)
    {
        static_cast<Triangle*>(elems[i])->draw();
    }

    glColor3f(1.0f, 1.0f, 1.0f);
    glDisable(GL_LIGHTING);
    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if (getPrevious()!=NULL && dynamic_cast<Abstract::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<Abstract::VisualModel*>(getPrevious())->draw();
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
void TriangleModel::computeContinuousBoundingBox (double dt)
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

    std::vector<Vector3> &verts = *(mmodel->getX());
    std::vector<Vector3> &velocityVerts = *(mmodel->getV());

    int size = verts.size();
    newVertices.reserve(size);

    for (int i = 0; i < size; i++)
    {
        Vector3 newPos = verts[i];
        newPos += velocityVerts[i] * dt; //getContext()->getDt();
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

    for (unsigned int i=0; i<elems.size(); i++)
        static_cast<Triangle*>(elems[i])->recalcContinuousBBox(dt);
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

    findBoundingBox(*(mmodel->getX()), minBB, maxBB);

    //std::cout << "BBox: <"<<minBB[0]<<','<<minBB[1]<<','<<minBB[2]<<">-<"<<maxBB[0]<<','<<maxBB[1]<<','<<maxBB[2]<<">\n";

    cubeModel->setCube(0,minBB, maxBB);

    for (unsigned int i=0; i<elems.size(); i++)
        static_cast<Triangle*>(elems[i])->recalcBBox();
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

