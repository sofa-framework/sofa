/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_COMPONENT_COLLISION_TETRAHEDRONMODEL_CPP
#include <SofaMiscCollision/TetrahedronModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CubeModel.h>
#include <sofa/helper/gl/template.h>
#include <sofa/simulation/Node.h>
#include <SofaBaseTopology/RegularGridTopology.h>
#include <sofa/core/CollisionElement.h>
#include <sofa/core/ObjectFactory.h>
#include <vector>
#include <sofa/helper/system/gl.h>
#include <iostream>
#include <SofaMeshCollision/BarycentricContactMapper.inl>
#include <sofa/helper/Factory.inl>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(TetrahedronModel)

int TetrahedronModelClass = core::RegisterObject("collision model using a tetrahedral mesh, as described in BaseMeshTopology")
        .add< TetrahedronModel >()
        .addAlias("Tetrahedron")
        ;

TetrahedronModel::TetrahedronModel()
    : tetra(NULL), mstate(NULL)
{
    enum_type = TETRAHEDRON_TYPE;
}

void TetrahedronModel::resize(int size)
{
    this->core::CollisionModel::resize(size);
    elems.resize(size);
    if (getPrevious() != NULL) getPrevious()->resize(0); // force recomputation of bounding tree
}

void TetrahedronModel::init()
{
    _topology = this->getContext()->getMeshTopology();

    this->CollisionModel::init();
    mstate = dynamic_cast< core::behavior::MechanicalState<Vec3Types>* > (getContext()->getMechanicalState());

    if (mstate==NULL)
    {
        serr<<"TetrahedronModel requires a Vec3 Mechanical Model" << sendl;
        return;
    }

    if (!_topology)
    {
        serr<<"TetrahedronModel requires a BaseMeshTopology" << sendl;
        return;
    }

    tetra = &_topology->getTetrahedra();
    resize(tetra->size());

}

void TetrahedronModel::handleTopologyChange()
{
    resize(_topology->getNbTetrahedra());
}

void TetrahedronModel::draw(const core::visual::VisualParams* vparams,int index)
{
#ifndef SOFA_NO_OPENGL
    Tetrahedron t(this,index);
    glBegin(GL_TRIANGLES);
    Coord p1 = t.p1();
    Coord p2 = t.p2();
    Coord p3 = t.p3();
    Coord p4 = t.p4();
    Coord c = (p1+p2+p3+p4)*0.25f;
    p1 += (c-p1)*0.1f;
    p2 += (c-p2)*0.1f;
    p3 += (c-p3)*0.1f;
    p4 += (c-p4)*0.1f;
    Coord n1,n2,n3,n4;
    n1 = cross(p3-p1,p2-p1); n1.normalize();
    helper::gl::glNormalT(n1);
    helper::gl::glVertexT(p1);
    helper::gl::glVertexT(p3);
    helper::gl::glVertexT(p2);

    n2 = cross(p4-p1,p3-p1); n2.normalize();
    helper::gl::glNormalT(n2);
    helper::gl::glVertexT(p1);
    helper::gl::glVertexT(p4);
    helper::gl::glVertexT(p3);

    n3 = cross(p2-p1,p4-p1); n3.normalize();
    helper::gl::glNormalT(n3);
    helper::gl::glVertexT(p1);
    helper::gl::glVertexT(p2);
    helper::gl::glVertexT(p4);

    n4 = cross(p3-p2,p4-p2); n4.normalize();
    helper::gl::glNormalT(n4);
    helper::gl::glVertexT(p2);
    helper::gl::glVertexT(p3);
    helper::gl::glVertexT(p4);
    glEnd();
    if (vparams->displayFlags().getShowNormals())
    {
        Coord p;
        glBegin(GL_LINES);
        p = (p1+p2+p3)*(1.0/3.0);
        helper::gl::glVertexT(p);
        helper::gl::glVertexT(p+n1*0.1);
        p = (p1+p3+p4)*(1.0/3.0);
        helper::gl::glVertexT(p);
        helper::gl::glVertexT(p+n2*0.1);
        p = (p1+p4+p2)*(1.0/3.0);
        helper::gl::glVertexT(p);
        helper::gl::glVertexT(p+n3*0.1);
        p = (p2+p3+p4)*(1.0/3.0);
        helper::gl::glVertexT(p);
        helper::gl::glVertexT(p+n4*0.1);
        glEnd();
    }
#endif /* SOFA_NO_OPENGL */
}

void TetrahedronModel::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (mstate && _topology && vparams->displayFlags().getShowCollisionModels())
    {
        if (vparams->displayFlags().getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glEnable(GL_LIGHTING);
        //Enable<GL_BLEND> blending;
        //glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

        glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, getColor4f());
        static const float emissive[4] = { 0.0f, 0.0f, 0.0f, 0.0f};
        static const float specular[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
        glMaterialfv (GL_FRONT_AND_BACK, GL_EMISSION, emissive);
        glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, specular);
        glMaterialf (GL_FRONT_AND_BACK, GL_SHININESS, 20);

        for (int i=0; i<size; i++)
        {
            draw(vparams,i);
        }

        glColor3f(1.0f, 1.0f, 1.0f);
        glDisable(GL_LIGHTING);
        if (vparams->displayFlags().getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    if (getPrevious()!=NULL && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);
#endif /* SOFA_NO_OPENGL */
}

void TetrahedronModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    if (!mstate || !_topology) return;
    if (!isMoving() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    Vector3 minElem, maxElem;
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    for (int i=0; i<size; i++)
    {
        Tetrahedron t(this,i);
        const Vector3& pt1 = x[t.p1Index()];
        const Vector3& pt2 = x[t.p2Index()];
        const Vector3& pt3 = x[t.p3Index()];
        const Vector3& pt4 = x[t.p4Index()];
        Matrix3 m, minv;
        m[0] = pt2-pt1;
        m[1] = pt3-pt1;
        m[2] = pt4-pt1;
        m.transpose();
        minv.invert(m);
        elems[i].coord0 = pt1;
        elems[i].bary2coord = m;
        elems[i].coord2bary = minv;
    }

    if (maxDepth == 0)
    {
        // no hierarchy
        if (empty())
            cubeModel->resize(0);
        else
        {
            cubeModel->resize(1);
            minElem = x[0];
            maxElem = x[0];
            for (unsigned i=1; i<x.size(); i++)
            {
                const Vector3& pt1 = x[i];
                if (pt1[0] > maxElem[0]) maxElem[0] = pt1[0];
                else if (pt1[0] < minElem[0]) minElem[0] = pt1[0];
                if (pt1[1] > maxElem[1]) maxElem[1] = pt1[1];
                else if (pt1[1] < minElem[1]) minElem[1] = pt1[1];
                if (pt1[2] > maxElem[2]) maxElem[2] = pt1[2];
                else if (pt1[2] < minElem[2]) minElem[2] = pt1[2];
            }
            cubeModel->setLeafCube(0, std::make_pair(this->begin(),this->end()), minElem, maxElem); // define the bounding box of the current Tetrahedron
        }
    }
    else
    {
        cubeModel->resize(size);  // size = number of Tetrahedrons
        if (!empty())
        {
            for (int i=0; i<size; i++)
            {
                Tetrahedron t(this,i);
                const Vector3& pt1 = x[t.p1Index()];
                const Vector3& pt2 = x[t.p2Index()];
                const Vector3& pt3 = x[t.p3Index()];
                const Vector3& pt4 = x[t.p4Index()];
                for (int c = 0; c < 3; c++)
                {
                    minElem[c] = pt1[c];
                    maxElem[c] = pt1[c];
                    if (pt2[c] > maxElem[c]) maxElem[c] = pt2[c];
                    else if (pt2[c] < minElem[c]) minElem[c] = pt2[c];
                    if (pt3[c] > maxElem[c]) maxElem[c] = pt3[c];
                    else if (pt3[c] < minElem[c]) minElem[c] = pt3[c];
                    if (pt4[c] > maxElem[c]) maxElem[c] = pt4[c];
                    else if (pt4[c] < minElem[c]) minElem[c] = pt4[c];
                }
                cubeModel->setParentOf(i, minElem, maxElem); // define the bounding box of the current Tetrahedron
            }
            cubeModel->computeBoundingTree(maxDepth);
        }
    }
}

ContactMapperCreator< ContactMapper<TetrahedronModel> > TetrahedronContactMapperClass("default",true);

template class SOFA_MISC_COLLISION_API ContactMapper<TetrahedronModel, sofa::defaulttype::Vec3Types>;

} // namespace collision

} // namespace component

} // namespace sofa
