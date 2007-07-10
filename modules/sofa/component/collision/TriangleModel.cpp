/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/component/collision/Triangle.h>
#include <sofa/core/CollisionElement.h>
#include <sofa/core/ObjectFactory.h>
#include <vector>
#include <GL/gl.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(Triangle)

int TriangleModelClass = core::RegisterObject("collision model using a triangular mesh")
        .add< TriangleModel >()
        .addAlias("Triangle")
        ;

TriangleModel::TriangleModel()
    : meshRevision(-1), mstate(NULL), mesh(NULL)
{
}

void TriangleModel::resize(int size)
{
    this->core::CollisionModel::resize(size);
    elems.resize(size);
}

void TriangleModel::init()
{
    this->CollisionModel::init();
    mstate = dynamic_cast< core::componentmodel::behavior::MechanicalState<Vec3Types>* > (getContext()->getMechanicalState());

    if (mstate==NULL)
    {
        std::cerr << "ERROR: TriangleModel requires a Vec3 Mechanical Model.\n";
        return;
    }

    sofa::core::componentmodel::topology::BaseTopology* bt = dynamic_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());
    if (bt)
    {
        sofa::core::componentmodel::topology::TopologyContainer *container=bt->getTopologyContainer();
        sofa::component::topology::TriangleSetTopologyContainer *tstc= dynamic_cast<sofa::component::topology::TriangleSetTopologyContainer *>(container);
        if (tstc)
        {
            const std::vector<sofa::component::topology::Triangle> &ta=tstc->getTriangleArray();
            resize(ta.size());
            for(unsigned int i=0; i<ta.size(); ++i)
            {
                elems[i].i1=ta[i][0];
                elems[i].i2=ta[i][1];
                elems[i].i3=ta[i][2];
            }
        }
        else
        {
            std::cerr << "ERROR: Topology is not a TriangleSetTopology.\n";
            return;
        }
    }
    else
    {
        mesh = dynamic_cast< topology::MeshTopology* > (getContext()->getTopology());

        if (mesh==NULL)
        {
            std::cerr << "ERROR: TriangleModel requires a Mesh Topology.\n";
            return;
        }
        updateFromTopology();
    }
    for (int i=0; i<size; i++)
    {
        Triangle t(this,i);
        const Vector3& pt1 = t.p1();
        const Vector3& pt2 = t.p2();
        const Vector3& pt3 = t.p3();

        t.n() = cross(pt2-pt1,pt3-pt1);
        t.n().normalize();
    }
}

void TriangleModel::updateFromTopology()
{
    const int npoints = mstate->getX()->size();
    const int ntris = mesh->getNbTriangles();
    const int nquads = mesh->getNbQuads();
    const int newsize = ntris+2*nquads;
    needsUpdate=true;

    int revision = mesh->getRevision();
    if (revision == meshRevision && newsize==size)
    {
        needsUpdate=false;
        return;
    }

    resize(newsize);
    int index = 0;
    //VecCoord& x = *mstate->getX();
    //VecDeriv& v = *mstate->getV();
    vector<bool> pflags(npoints);
    std::set<std::pair<int,int> > eflags;
    for (int i=0; i<ntris; i++)
    {
        topology::MeshTopology::Triangle idx = mesh->getTriangle(i);
        if (idx[0] >= npoints || idx[1] >= npoints || idx[2] >= npoints)
        {
            std::cerr << "ERROR: Out of range index in triangle "<<i<<": "<<idx[0]<<" "<<idx[1]<<" "<<idx[2]<<" ( total points="<<npoints<<")\n";
            continue;
        }
        elems[index].i1 = idx[0];
        elems[index].i2 = idx[1];
        elems[index].i3 = idx[2];
        int f = 0;
        if (!pflags[elems[index].i1])
        {
            f |= FLAG_P1;
            pflags[elems[index].i1] = true;
        }
        if (!pflags[elems[index].i2])
        {
            f |= FLAG_P2;
            pflags[elems[index].i2] = true;
        }
        if (!pflags[elems[index].i3])
        {
            f |= FLAG_P3;
            pflags[elems[index].i3] = true;
        }
        if (eflags.insert( (elems[index].i1<elems[index].i2)?std::make_pair(elems[index].i1,elems[index].i2):std::make_pair(elems[index].i2,elems[index].i1) ).second)
        {
            f |= FLAG_E12;
        }
        if (eflags.insert( (elems[index].i2<elems[index].i3)?std::make_pair(elems[index].i2,elems[index].i3):std::make_pair(elems[index].i3,elems[index].i2) ).second)
        {
            f |= FLAG_E23;
        }
        if (eflags.insert( (elems[index].i3<elems[index].i1)?std::make_pair(elems[index].i3,elems[index].i1):std::make_pair(elems[index].i1,elems[index].i3) ).second)
        {
            f |= FLAG_E31;
        }
        elems[index].flags = f;

        //elems[index].i3 = idx[2];
        ++index;
    }
    for (int i=0; i<nquads; i++)
    {
        topology::MeshTopology::Quad idx = mesh->getQuad(i);
        if (idx[0] >= npoints || idx[1] >= npoints || idx[2] >= npoints || idx[3] >= npoints)
        {
            std::cerr << "ERROR: Out of range index in quad "<<i<<": "<<idx[0]<<" "<<idx[1]<<" "<<idx[2]<<" "<<idx[3]<<" ( total points="<<npoints<<")\n";
            continue;
        }
        elems[index].i1 = idx[0];
        elems[index].i2 = idx[1];
        elems[index].i3 = idx[2];
        int f = 0;
        if (!pflags[elems[index].i1])
        {
            f |= FLAG_P1;
            pflags[elems[index].i1] = true;
        }
        if (!pflags[elems[index].i2])
        {
            f |= FLAG_P2;
            pflags[elems[index].i2] = true;
        }
        if (!pflags[elems[index].i3])
        {
            f |= FLAG_P3;
            pflags[elems[index].i3] = true;
        }
        if (eflags.insert( (elems[index].i1<elems[index].i2)?std::make_pair(elems[index].i1,elems[index].i2):std::make_pair(elems[index].i2,elems[index].i1) ).second)
        {
            f |= FLAG_E12;
        }
        if (eflags.insert( (elems[index].i2<elems[index].i3)?std::make_pair(elems[index].i2,elems[index].i3):std::make_pair(elems[index].i3,elems[index].i2) ).second)
        {
            f |= FLAG_E23;
        }
        elems[index].flags = f;

        ++index;
        elems[index].i1 = idx[0];
        elems[index].i2 = idx[2];
        elems[index].i3 = idx[3];
        f = 0;
        if (!pflags[elems[index].i3])
        {
            f |= FLAG_P3;
            pflags[elems[index].i3] = true;
        }
        if (eflags.insert( (elems[index].i2<elems[index].i3)?std::make_pair(elems[index].i2,elems[index].i3):std::make_pair(elems[index].i3,elems[index].i2) ).second)
        {
            f |= FLAG_E23;
        }
        if (eflags.insert( (elems[index].i3<elems[index].i1)?std::make_pair(elems[index].i3,elems[index].i1):std::make_pair(elems[index].i1,elems[index].i3) ).second)
        {
            f |= FLAG_E31;
        }
        elems[index].flags = f;

        ++index;
    }
    meshRevision = revision;
}

void TriangleModel::draw(int index)
{
    Triangle t(this,index);
    glBegin(GL_TRIANGLES);
    glNormal3dv(t.n().ptr());
    glVertex3dv(t.p1().ptr());
    glVertex3dv(t.p2().ptr());
    glVertex3dv(t.p3().ptr());
    glEnd();
}
void TriangleModel::handleTopologyChange()
{
    sofa::core::componentmodel::topology::BaseTopology* bt = dynamic_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());
    if (bt)
    {

        std::list<const sofa::core::componentmodel::topology::TopologyChange *>::const_iterator itBegin=bt->firstChange();
        std::list<const sofa::core::componentmodel::topology::TopologyChange *>::const_iterator itEnd=bt->lastChange();

        while( itBegin != itEnd )
        {
            core::componentmodel::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();
            // Since we are using identifier, we can safely use C type casts.
            switch( changeType )
            {

            case core::componentmodel::topology::TRIANGLESADDED:
            {
                TriangleInfo t;
                const sofa::component::topology::TrianglesAdded *ta=dynamic_cast< const sofa::component::topology::TrianglesAdded * >( *itBegin );
                for (unsigned int i=0; i<ta->getNbAddedTriangles(); ++i)
                {
                    t.i1=(int)(ta->triangleArray[i])[0];
                    t.i2=(int)(ta->triangleArray[i])[1];
                    t.i3=(int)(ta->triangleArray[i])[2];
                    elems.push_back(t);
                }
                needsUpdate=true;
                break;
            }

            case core::componentmodel::topology::TRIANGLESREMOVED:
            {
                const std::vector<unsigned int> &tab = ( dynamic_cast< const sofa::component::topology::TrianglesRemoved *>( *itBegin ) )->getArray();
                unsigned int  last= elems.size() -1;
                TriangleInfo tmp;
                for (unsigned int i = 0; i <tab.size(); ++i)
                {
                    tmp = elems[tab[i]];
                    elems[tab[i]] = elems[last];
                    elems[last] = tmp;

                    --last;
                }
                resize( elems.size() - tab.size() );
                needsUpdate=true;
                break;
            }
            case core::componentmodel::topology::POINTSREMOVED:
            {
                sofa::core::componentmodel::topology::TopologyContainer *container=bt->getTopologyContainer();
                sofa::component::topology::TriangleSetTopologyContainer *tstc= dynamic_cast<sofa::component::topology::TriangleSetTopologyContainer *>(container);
                if (tstc)
                {
                    const std::vector< std::vector<unsigned int> > &tvsa=tstc->getTriangleVertexShellArray();
                    unsigned int last = tvsa.size() -1;
                    unsigned int i,j;
                    const std::vector<unsigned int> tab = ( dynamic_cast< const sofa::component::topology::PointsRemoved * >( *itBegin ) )->getArray();
                    for ( i = 0; i < tab.size(); ++i)
                    {
                        const std::vector<unsigned int> &shell=tvsa[last];
                        for (j=0; j<shell.size(); ++j)
                        {
                            if (elems[shell[j]].i1==last)
                                elems[shell[j]].i1=tab[i];
                            else if (elems[shell[j]].i2==last)
                                elems[shell[j]].i2=tab[i];
                            else if (elems[shell[j]].i3==last)
                                elems[shell[j]].i3=tab[i];
                        }
                        --last;
                    }
                }
                break;
            }
            default:
                // Ignore events that are not Triangle  related.
                break;
            }; // switch( changeType )

            ++itBegin;
        } // while( changeIt != last; )
    }
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
    if (isActive() && getPrevious()!=NULL && getContext()->getShowBoundingCollisionModels() && dynamic_cast<core::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<core::VisualModel*>(getPrevious())->draw();
}

void TriangleModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    if (mesh)
        updateFromTopology();
    if (needsUpdate && !cubeModel->empty()) cubeModel->resize(0);
    if (isStatic() && !cubeModel->empty() && !needsUpdate) return; // No need to recompute BBox if immobile

    needsUpdate=false;
    Vector3 minElem, maxElem;

    cubeModel->resize(size);  // size = number of triangles
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

            cubeModel->setParentOf(i, minElem, maxElem); // define the bounding box of the current triangle
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}
unsigned int TriangleModel::getNbTriangles() const
{
    if (mesh)
        return size;
    else
        mesh->getNbTriangles();
}
void TriangleModel::computeContinuousBoundingTree(double dt, int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    if (mesh)
        updateFromTopology();
    if (needsUpdate) cubeModel->resize(0);
    if (isStatic() && !cubeModel->empty() && !needsUpdate) return; // No need to recompute BBox if immobile

    needsUpdate=false;
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

void TriangleModel::buildOctree()
{
    cerr<<"TriangleModel::buildOctree(), coords = "<<*mstate->getX()<<endl;
    const int ntris = mesh->getNbTriangles();
    const int nquads = mesh->getNbQuads();
    for (int i=0; i<ntris; i++)
    {
        topology::MeshTopology::Triangle idx = mesh->getTriangle(i);
        cerr<<"  triangle "<< idx[0] <<", "<<idx[1]<<", "<<idx[2]<<endl;;
    }
    for (int i=0; i<nquads; i++)
    {
        topology::MeshTopology::Quad idx = mesh->getQuad(i);
        cerr<<"  triangle "<< idx[0] <<", "<<idx[1]<<", "<<idx[2]<<endl;;
        cerr<<"  triangle "<< idx[0] <<", "<<idx[2]<<", "<<idx[3]<<endl;;
    }
}

} // namespace collision

} // namespace component

} // namespace sofa

