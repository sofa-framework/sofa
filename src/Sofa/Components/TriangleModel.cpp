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
    : static_(false), mmodel(NULL), mesh(NULL)
{
}

void TriangleModel::resize(int size)
{
    this->Abstract::CollisionModel::resize(size);
    elems.resize(size);
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
    const int npoints = mmodel->getX()->size();
    const int ntris = mesh->getNbTriangles();
    const int nquads = mesh->getNbQuads();
    resize(ntris+2*nquads);
    int index = 0;
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
        elems[index].i1 = idx[0];
        elems[index].i2 = idx[1];
        elems[index].i3 = idx[2];
        ++index;
    }
    for (int i=0; i<nquads; i++)
    {
        MeshTopology::Quad idx = mesh->getQuad(i);
        if (idx[0] >= npoints || idx[1] >= npoints || idx[2] >= npoints || idx[3] >= npoints)
        {
            std::cerr << "ERROR: Out of range index in quad "<<i<<": "<<idx[0]<<" "<<idx[1]<<" "<<idx[2]<<" "<<idx[3]<<" ( total points="<<npoints<<")\n";
            continue;
        }
        elems[index].i1 = idx[0];
        elems[index].i2 = idx[1];
        elems[index].i3 = idx[2];
        ++index;
        elems[index].i1 = idx[0];
        elems[index].i2 = idx[2];
        elems[index].i3 = idx[3];
        ++index;
    }
}

void TriangleModel::draw(int index)
{
    Triangle t(this,index);
    glBegin(GL_TRIANGLES);
    glNormal3dv(t.n());
    glVertex3dv(t.p1());
    glVertex3dv(t.p2());
    glVertex3dv(t.p3());
    glEnd();
}

void TriangleModel::draw()
{
    if (isActive() && getContext()->getShowCollisionModels())
    {
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glEnable(GL_LIGHTING);
        //Enable<GL_BLEND> blending;
        //glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

        static const float color[4] = { 1.0f, 0.2f, 0.0f, 1.0f};
        static const float colorStatic[4] = { 0.5f, 0.5f, 0.5f, 1.0f};
        if (isStatic())
            glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, colorStatic);
        else
            glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
        static const float emissive[4] = { 0.0f, 0.0f, 0.0f, 0.0f};
        static const float specular[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
        glMaterialfv (GL_FRONT_AND_BACK, GL_EMISSION, emissive);
        glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, specular);
        glMaterialf (GL_FRONT_AND_BACK, GL_SHININESS, 20);

        for (int i=0; i<size; i++)
        {
            draw(i);
        }

        glColor3f(1.0f, 1.0f, 1.0f);
        glDisable(GL_LIGHTING);
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    if (isActive() && getPrevious()!=NULL && getContext()->getShowBoundingCollisionModels() && dynamic_cast<Abstract::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<Abstract::VisualModel*>(getPrevious())->draw();
}

void TriangleModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    if (isStatic() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        for (int i=0; i<size; i++)
        {
            Triangle t(this,i);
            const Vector3& pt1 = t.p1();
            const Vector3& pt2 = t.p2();
            const Vector3& pt3 = t.p3();

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt1[c];
                maxElem[c] = pt1[c];
                if (pt2[c] > maxElem[c]) maxElem[c] = pt2[c];
                else if (pt2[c] < minElem[c]) minElem[c] = pt2[c];
                if (pt3[c] > maxElem[c]) maxElem[c] = pt3[c];
                else if (pt3[c] < minElem[c]) minElem[c] = pt3[c];
            }

            // Also recompute normal vector
            t.n() = cross(pt2-pt1,pt3-pt1);
            t.n().normalize();

            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

void TriangleModel::computeContinuousBoundingTree(double dt, int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    if (isStatic() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        for (int i=0; i<size; i++)
        {
            Triangle t(this,i);
            const Vector3& pt1 = t.p1();
            const Vector3& pt2 = t.p2();
            const Vector3& pt3 = t.p3();
            const Vector3 pt1v = pt1 + t.v1()*dt;
            const Vector3 pt2v = pt2 + t.v2()*dt;
            const Vector3 pt3v = pt3 + t.v3()*dt;

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt1[c];
                maxElem[c] = pt1[c];
                if (pt2[c] > maxElem[c]) maxElem[c] = pt2[c];
                else if (pt2[c] < minElem[c]) minElem[c] = pt2[c];
                if (pt3[c] > maxElem[c]) maxElem[c] = pt3[c];
                else if (pt3[c] < minElem[c]) minElem[c] = pt3[c];

                if (pt1v[c] > maxElem[c]) maxElem[c] = pt1v[c];
                else if (pt1v[c] < minElem[c]) minElem[c] = pt1v[c];
                if (pt2v[c] > maxElem[c]) maxElem[c] = pt2v[c];
                else if (pt2v[c] < minElem[c]) minElem[c] = pt2v[c];
                if (pt3v[c] > maxElem[c]) maxElem[c] = pt3v[c];
                else if (pt3v[c] < minElem[c]) minElem[c] = pt3v[c];
            }

            // Also recompute normal vector
            t.n() = cross(pt2-pt1,pt3-pt1);
            t.n().normalize();

            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

} // namespace Components

} // namespace Sofa

