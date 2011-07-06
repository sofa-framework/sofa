/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/system/config.h>
#include <sofa/helper/proximity.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/collision/Intersection.inl>
#include <iostream>
#include <algorithm>




#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/PointLocalMinDistanceFilter.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/core/ObjectFactory.h>
#include <vector>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/gl/template.h>
#include <sofa/core/collision/Intersection.inl>

#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/simulation/common/Simulation.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace helper;

SOFA_DECL_CLASS(Point)

int PointModelClass = core::RegisterObject("Collision model which represents a set of points")
        .add< PointModel >()
        .addAlias("Point")
// .addAlias("PointModel")
        .addAlias("PointMesh")
        .addAlias("PointSet")
        ;

PointModel::PointModel()
    : bothSide(initData(&bothSide, false, "bothSide", "activate collision on both side of the point model (when surface normals are defined on these points)") )
    , mstate(NULL)
    , computeNormals( initData(&computeNormals, false, "computeNormals", "activate computation of normal vectors (required for some collision detection algorithms)") )
    , PointActiverEngine(initData(&PointActiverEngine,"PointActiverEngine", "path of a component PointActiver that activate or deactivate collision point during execution") )
    , m_lmdFilter( NULL )
    , m_displayFreePosition(initData(&m_displayFreePosition, false, "displayFreePosition", "Display Collision Model Points free position(in green)") )
{
}

void PointModel::resize(int size)
{
    this->core::CollisionModel::resize(size);
}

void PointModel::init()
{
    this->CollisionModel::init();
    mstate = dynamic_cast< core::behavior::MechanicalState<Vec3Types>* > (getContext()->getMechanicalState());

    if (mstate==NULL)
    {
        serr<<"ERROR: PointModel requires a Vec3 Mechanical Model" << sendl;
        return;
    }

    simulation::Node* node = dynamic_cast< simulation::Node* >(this->getContext());
    if (node != 0)
    {
        m_lmdFilter = node->getNodeObject< PointLocalMinDistanceFilter >();
    }

    const int npoints = mstate->getX()->size();
    resize(npoints);
    if (computeNormals.getValue()) updateNormals();



    const std::string path = PointActiverEngine.getValue();

    if (path.size()==0)
    {
        myActiver = new PointActiver();
//		std::cout<<"no Point Activer found path ="<<path<<std::endl;
    }
    else
    {
        this->getContext()->get(myActiver ,path  );

        if (myActiver==NULL)
        {
            myActiver = new PointActiver();
            std::cout<<"wrong path for Point Activer  "<<std::endl;
        }
        else
            std::cout<<"Point Activer found !"<<std::endl;
    }

}

void PointModel::draw(const core::visual::VisualParams* ,int index)
{
    Point p(this,index);
    if (!p.activated())
        return;
    glBegin(GL_POINTS);
    helper::gl::glVertexT(p.p());
    glEnd();
    if ((unsigned)index < normals.size())
    {
        glBegin(GL_LINES);
        helper::gl::glVertexT(p.p());
        helper::gl::glVertexT(p.p()+normals[index]*0.1f);
        glEnd();
    }
}

void PointModel::draw(const core::visual::VisualParams* vparams)
{
    if (getContext()->getShowCollisionModels())
    {
        if (getContext()->getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0,true);

        // Check topological modifications
        const int npoints = mstate->getX()->size();
        if (npoints != size)
        {
            resize(npoints);
        }

        std::vector< Vector3 > pointsP;
        std::vector< Vector3 > pointsL;
        for (int i = 0; i < size; i++)
        {
            Point p(this,i);
            if (p.activated())
            {
                pointsP.push_back(p.p());
                if ((unsigned)i < normals.size())
                {
                    pointsL.push_back(p.p());
                    pointsL.push_back(p.p()+normals[i]*0.1f);
                }
            }
        }

        vparams->drawTool()->drawPoints(pointsP, 3, Vec<4,float>(getColor4f()));
        vparams->drawTool()->drawLines(pointsL, 1, Vec<4,float>(getColor4f()));

        if (m_displayFreePosition.getValue())
        {
            std::vector< Vector3 > pointsPFree;

            for (int i = 0; i < size; i++)
            {
                Point p(this,i);
                if (p.activated())
                {
                    pointsPFree.push_back(p.pFree());
                }
            }

            vparams->drawTool()->drawPoints(pointsPFree, 3, Vec<4,float>(0.0f,1.0f,0.2f,1.0f));
        }

        if (getContext()->getShowWireFrame())
            vparams->drawTool()->setPolygonMode(0,false);
    }

    if (getPrevious()!=NULL && getContext()->getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);
}

bool PointModel::canCollideWithElement(int index, CollisionModel* model2, int index2)
{
    //sout<<"PointModel("<<this->getName()<<") :: canCollideWithElement("<<model2->getName()<<") is called"<<sendl;
    if (!this->bSelfCollision.getValue()) return true; // we need to perform this verification process only for the selfcollision case.
    if (this->getContext() != model2->getContext()) return true;


    //if(index==4)
    //{
    //	std::cout<<" model : "<<model2->getName()<<"at index ["<<index2<<"] can collide with point 4 ?"<<std::endl;
    //}

    if (model2 == this)
    {

        if (index==index2)
            return false;

        sofa::core::topology::BaseMeshTopology* topology = this->getMeshTopology();



        // in the neighborhood, if we find a point in common, we cancel the collision
        const helper::vector <unsigned int>& verticesAroundVertex1 =topology->getVerticesAroundVertex(index);
        const helper::vector <unsigned int>& verticesAroundVertex2 =topology->getVerticesAroundVertex(index2);

        for (unsigned int i1=0; i1<verticesAroundVertex1.size(); i1++)
        {
            unsigned int v1 = verticesAroundVertex1[i1];

            for (unsigned int i2=0; i2<verticesAroundVertex2.size(); i2++)
            {
                if (v1==verticesAroundVertex2[i2])
                    return false;
            }
        }
        return true; // || index > index2+1;
    }
    else
        return model2->canCollideWithElement(index2, this, index);
}

void PointModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int npoints = mstate->getX()->size();
    bool updated = false;
    if (npoints != size)
    {
        resize(npoints);
        updated = true;
    }
    if (updated) cubeModel->resize(0);
    if (!isMoving() && !cubeModel->empty() && !updated) return; // No need to recompute BBox if immobile

    if (computeNormals.getValue()) updateNormals();

    cubeModel->resize(size);
    if (!empty())
    {
        //VecCoord& x = *mstate->getX();
        for (int i=0; i<size; i++)
        {
            Point p(this,i);
            const Vector3& pt = p.p();
            cubeModel->setParentOf(i, pt, pt);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }

    if (m_lmdFilter != 0)
    {
        m_lmdFilter->invalidate();
    }
}

void PointModel::computeContinuousBoundingTree(double dt, int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int npoints = mstate->getX()->size();
    bool updated = false;
    if (npoints != size)
    {
        resize(npoints);
        updated = true;
    }
    if (!isMoving() && !cubeModel->empty() && !updated) return; // No need to recompute BBox if immobile

    if (computeNormals.getValue()) updateNormals();

    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        //VecCoord& x = *mstate->getX();
        //VecDeriv& v = *mstate->getV();
        for (int i=0; i<size; i++)
        {
            Point p(this,i);
            const Vector3& pt = p.p();
            const Vector3 ptv = pt + p.v()*dt;

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt[c];
                maxElem[c] = pt[c];
                if (ptv[c] > maxElem[c]) maxElem[c] = ptv[c];
                else if (ptv[c] < minElem[c]) minElem[c] = ptv[c];
            }
            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

void PointModel::updateNormals()
{
    const VecCoord& x = *mstate->getX();
    int n = x.size();
    normals.resize(n);
    for (int i=0; i<n; ++i)
    {
        normals[i].clear();
    }
    core::topology::BaseMeshTopology* mesh = getContext()->getMeshTopology();
    if (mesh->getNbTetrahedra()+mesh->getNbHexahedra() > 0)
    {
        if (mesh->getNbTetrahedra()>0)
        {
            const core::topology::BaseMeshTopology::SeqTetrahedra &elems = mesh->getTetrahedra();
            for (unsigned int i=0; i < elems.size(); ++i)
            {
                const core::topology::BaseMeshTopology::Tetra &e = elems[i];
                const Coord& p1 = x[e[0]];
                const Coord& p2 = x[e[1]];
                const Coord& p3 = x[e[2]];
                const Coord& p4 = x[e[3]];
                Coord& n1 = normals[e[0]];
                Coord& n2 = normals[e[1]];
                Coord& n3 = normals[e[2]];
                Coord& n4 = normals[e[3]];
                Coord n;
                n = cross(p3-p1,p2-p1); n.normalize();
                n1 += n;
                n2 += n;
                n3 += n;
                n = cross(p4-p1,p3-p1); n.normalize();
                n1 += n;
                n3 += n;
                n4 += n;
                n = cross(p2-p1,p4-p1); n.normalize();
                n1 += n;
                n4 += n;
                n2 += n;
                n = cross(p3-p2,p4-p2); n.normalize();
                n2 += n;
                n4 += n;
                n3 += n;
            }
        }
        /// @TODO Hexahedra
    }
    else if (mesh->getNbTriangles()+mesh->getNbQuads() > 0)
    {
        if (mesh->getNbTriangles()>0)
        {
            const core::topology::BaseMeshTopology::SeqTriangles &elems = mesh->getTriangles();
            for (unsigned int i=0; i < elems.size(); ++i)
            {
                const core::topology::BaseMeshTopology::Triangle &e = elems[i];
                const Coord& p1 = x[e[0]];
                const Coord& p2 = x[e[1]];
                const Coord& p3 = x[e[2]];
                Coord& n1 = normals[e[0]];
                Coord& n2 = normals[e[1]];
                Coord& n3 = normals[e[2]];
                Coord n;
                n = cross(p2-p1,p3-p1); n.normalize();
                n1 += n;
                n2 += n;
                n3 += n;
            }
        }
        if (mesh->getNbQuads()>0)
        {
            const core::topology::BaseMeshTopology::SeqQuads &elems = mesh->getQuads();
            for (unsigned int i=0; i < elems.size(); ++i)
            {
                const core::topology::BaseMeshTopology::Quad &e = elems[i];
                const Coord& p1 = x[e[0]];
                const Coord& p2 = x[e[1]];
                const Coord& p3 = x[e[2]];
                const Coord& p4 = x[e[3]];
                Coord& n1 = normals[e[0]];
                Coord& n2 = normals[e[1]];
                Coord& n3 = normals[e[2]];
                Coord& n4 = normals[e[3]];
                Coord n;
                n = cross(p3-p1,p4-p2); n.normalize();
                n1 += n;
                n2 += n;
                n3 += n;
                n4 += n;
            }
        }
    }
    for (int i=0; i<n; ++i)
    {
        SReal l = normals[i].norm();
        if (l > 1.0e-3)
            normals[i] *= 1/l;
        else
            normals[i].clear();
    }
}


bool Point::testLMD(const Vector3 &PQ, double &coneFactor, double &coneExtension)
{

    Vector3 pt = p();

    sofa::core::topology::BaseMeshTopology* mesh = model->getMeshTopology();
    helper::vector<Vector3> x = (*model->mstate->getX());

    const helper::vector <unsigned int>& trianglesAroundVertex = mesh->getTrianglesAroundVertex(index);
    const helper::vector <unsigned int>& edgesAroundVertex = mesh->getEdgesAroundVertex(index);


    Vector3 nMean;

    for (unsigned int i=0; i<trianglesAroundVertex.size(); i++)
    {
        unsigned int t = trianglesAroundVertex[i];
        const fixed_array<unsigned int,3>& ptr = mesh->getTriangle(t);
        Vector3 nCur = (x[ptr[1]]-x[ptr[0]]).cross(x[ptr[2]]-x[ptr[0]]);
        nCur.normalize();
        nMean += nCur;
    }

    if (trianglesAroundVertex.size()==0)
    {
        for (unsigned int i=0; i<edgesAroundVertex.size(); i++)
        {
            unsigned int e = edgesAroundVertex[i];
            const fixed_array<unsigned int,2>& ped = mesh->getEdge(e);
            Vector3 l = (pt - x[ped[0]]) + (pt - x[ped[1]]);
            l.normalize();
            nMean += l;
        }
    }

    if (nMean.norm()> 0.0000000001)
        nMean.normalize();


    for (unsigned int i=0; i<edgesAroundVertex.size(); i++)
    {
        unsigned int e = edgesAroundVertex[i];
        const fixed_array<unsigned int,2>& ped = mesh->getEdge(e);
        Vector3 l = (pt - x[ped[0]]) + (pt - x[ped[1]]);
        l.normalize();
        double computedAngleCone = dot(nMean , l) * coneFactor;
        if (computedAngleCone<0)
            computedAngleCone=0.0;
        computedAngleCone+=coneExtension;
        if (dot(l , PQ) < -computedAngleCone*PQ.norm())
            return false;
    }
    return true;


}


PointLocalMinDistanceFilter *PointModel::getFilter() const
{
    return m_lmdFilter;
}


void PointModel::setFilter(PointLocalMinDistanceFilter *lmdFilter)
{
    m_lmdFilter = lmdFilter;
}


} // namespace collision

} // namespace component

} // namespace sofa

