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
#include <sofa/simulation/tree/GNode.h>
#include <sofa/component/topology/RegularGridTopology.h>
#include <sofa/core/CollisionElement.h>
#include <sofa/core/ObjectFactory.h>
#include <vector>
#include <sofa/helper/gl/template.h>
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

int TriangleModelClass = core::RegisterObject("collision model using a triangular mesh, as described in BaseMeshTopology")
        .add< TriangleModel >()
        .addAlias("TriangleMeshModel")
        .addAlias("TriangleSetModel")
        .addAlias("TriangleMesh")
        .addAlias("TriangleSet")
        .addAlias("Triangle")
        ;

TriangleModel::TriangleModel()
    : mstate(NULL)
    , computeNormals(initData(&computeNormals, true, "computeNormals", "set to false to disable computation of triangles normal"))
    , meshRevision(-1)
    , topology(NULL)
{
    triangles = &mytriangles;
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

    topology = dynamic_cast<Topology *>(getContext()->getTopology());
    if (!topology)
    {
        std::cerr << "ERROR: TriangleModel requires a BaseMeshTopology.\n";
        return;
    }

    setTopology = dynamic_cast<SetTopology*>(getContext()->getMainTopology());
    if (setTopology)
    {
        //std::cout << "INFO_print : Col - init TRIANGLE " << std::endl;
        sofa::component::topology::TriangleSetTopologyContainer *tstc= setTopology->getTriangleSetTopologyContainer();
        const sofa::helper::vector<sofa::component::topology::Triangle> &ta=tstc->getTriangleArray();
        triangles = &ta;
        resize(ta.size());
    }
    updateFromTopology();
    updateNormals();
}

void TriangleModel::updateNormals()
{
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

//    needsUpdate = false;
    const unsigned npoints = mstate->getX()->size();
    const unsigned ntris = topology->getNbTriangles();
    const unsigned nquads = topology->getNbQuads();
    const unsigned newsize = ntris+2*nquads;

    int revision = topology->getRevision();
    if (revision == meshRevision && newsize==(unsigned)size)
    {
        return;
    }
    needsUpdate=true;

    resize(newsize);

    if (newsize == ntris)
    {
        // no need to copy the triangle indices
        triangles = & topology->getTriangles();
    }
    else
    {
        triangles = &mytriangles;
        mytriangles.resize(newsize);
        int index = 0;
        for (unsigned i=0; i<ntris; i++)
        {
            topology::BaseMeshTopology::Triangle idx = topology->getTriangle(i);
            if (idx[0] >= npoints || idx[1] >= npoints || idx[2] >= npoints)
            {
                std::cerr << "ERROR: Out of range index in triangle "<<i<<": "<<idx[0]<<" "<<idx[1]<<" "<<idx[2]<<" ( total points="<<npoints<<")\n";
                if (idx[0] >= npoints) idx[0] = npoints-1;
                if (idx[1] >= npoints) idx[1] = npoints-1;
                if (idx[2] >= npoints) idx[2] = npoints-1;
            }
            mytriangles[index] = idx;
            ++index;
        }
        for (unsigned i=0; i<nquads; i++)
        {
            topology::BaseMeshTopology::Quad idx = topology->getQuad(i);
            if (idx[0] >= npoints || idx[1] >= npoints || idx[2] >= npoints || idx[3] >= npoints)
            {
                std::cerr << "ERROR: Out of range index in quad "<<i<<": "<<idx[0]<<" "<<idx[1]<<" "<<idx[2]<<" "<<idx[3]<<" ( total points="<<npoints<<")\n";
                if (idx[0] >= npoints) idx[0] = npoints-1;
                if (idx[1] >= npoints) idx[1] = npoints-1;
                if (idx[2] >= npoints) idx[2] = npoints-1;
                if (idx[3] >= npoints) idx[3] = npoints-1;
            }
            mytriangles[index][0] = idx[1];
            mytriangles[index][1] = idx[2];
            mytriangles[index][2] = idx[0];
            ++index;
            mytriangles[index][0] = idx[3];
            mytriangles[index][1] = idx[0];
            mytriangles[index][2] = idx[2];
            ++index;
        }
    }
    updateFlags();
    meshRevision = revision;
}

void TriangleModel::updateFlags(int ntri)
{
    if (ntri < 0) ntri = triangles->size();
    //VecCoord& x = *mstate->getX();
    //VecDeriv& v = *mstate->getV();
    vector<bool> pflags(mstate->getSize());
    std::set<std::pair<int,int> > eflags;
    for (unsigned i=0; i<triangles->size(); i++)
    {
        int f = 0;
        topology::Triangle t = (*triangles)[i];
        if (!pflags[t[0]])
        {
            f |= FLAG_P1;
            pflags[t[0]] = true;
        }
        if (!pflags[t[1]])
        {
            f |= FLAG_P2;
            pflags[t[1]] = true;
        }
        if (!pflags[t[2]])
        {
            f |= FLAG_P3;
            pflags[t[2]] = true;
        }
        if (eflags.insert( (t[0]<t[1])?std::make_pair(t[0],t[1]):std::make_pair(t[1],t[0]) ).second)
        {
            f |= FLAG_E12;
        }
        if (i < (unsigned)ntri && eflags.insert( (t[1]<t[2])?std::make_pair(t[1],t[2]):std::make_pair(t[2],t[1]) ).second) // don't use the diagonal edge of quads
        {
            f |= FLAG_E23;
        }
        if (eflags.insert( (t[2]<t[0])?std::make_pair(t[2],t[0]):std::make_pair(t[0],t[2]) ).second)
        {
            f |= FLAG_E31;
        }
        elems[i].flags = f;
    }
}

void TriangleModel::handleTopologyChange()
{
    bool debug_mode = false;

    if (triangles != &mytriangles)
    {
        // We use the same triangle array as the topology -> only resize and recompute flags
        resize(setTopology->getTriangleSetTopologyContainer()->getNumberOfTriangles());
        needsUpdate = true;
        updateFlags();
        updateNormals(); // not strictly necessary but useful if we display the model before the next collision iteration

        return;
    }

    sofa::core::componentmodel::topology::BaseTopology* bt = setTopology;
    if (bt)
    {

        std::list<const sofa::core::componentmodel::topology::TopologyChange *>::const_iterator itBegin=bt->firstChange();
        std::list<const sofa::core::componentmodel::topology::TopologyChange *>::const_iterator itEnd=bt->lastChange();


        while( itBegin != itEnd )
        {
            core::componentmodel::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();
            // Since we are using identifier, we can safely use C type casts.

            sofa::core::componentmodel::topology::TopologyContainer *container=bt->getTopologyContainer();

            sofa::component::topology::TriangleSetTopologyContainer *tstc= dynamic_cast<sofa::component::topology::TriangleSetTopologyContainer *>(container);

            switch( changeType )
            {


            case core::componentmodel::topology::ENDING_EVENT:
            {
                //std::cout << "INFO_print : Col - ENDING_EVENT" << std::endl;
                needsUpdate=true;
                break;
            }


            case core::componentmodel::topology::TRIANGLESADDED:
            {
                //std::cout << "INFO_print : Col - TRIANGLESADDED" << std::endl;
                TriangleInfo t;
                const sofa::component::topology::TrianglesAdded *ta=dynamic_cast< const sofa::component::topology::TrianglesAdded * >( *itBegin );
                for (unsigned int i=0; i<ta->getNbAddedTriangles(); ++i)
                {
                    mytriangles.push_back(ta->triangleArray[i]);
                }
                resize( mytriangles.size());
                needsUpdate=true;

                break;
            }

            case core::componentmodel::topology::TRIANGLESREMOVED:
            {
                //std::cout << "INFO_print : Col - TRIANGLESREMOVED" << std::endl;
                unsigned int last;
                unsigned int ind_last;

                if(tstc)
                {
                    last= (tstc->getTriangleArray()).size() - 1;
                }
                else
                {
                    last= elems.size() -1;
                }

                const sofa::helper::vector<unsigned int> &tab = ( dynamic_cast< const sofa::component::topology::TrianglesRemoved *>( *itBegin ) )->getArray();

                TriangleInfo tmp;
                topology::Triangle tmp2;

                for (unsigned int i = 0; i <tab.size(); ++i)
                {

                    unsigned int ind_k = tab[i];

                    tmp = elems[ind_k];
                    elems[ind_k] = elems[last];
                    elems[last] = tmp;

                    tmp2 = mytriangles[ind_k];
                    mytriangles[ind_k] = mytriangles[last];
                    mytriangles[last] = tmp2;

                    ind_last = elems.size() - 1;

                    if(last != ind_last)
                    {

                        tmp = elems[last];
                        elems[last] = elems[ind_last];
                        elems[ind_last] = tmp;

                        tmp2 = mytriangles[last];
                        mytriangles[last] = mytriangles[ind_last];
                        mytriangles[ind_last] = tmp2;
                    }

                    mytriangles.resize( elems.size() - 1 );
                    resize( elems.size() - 1 );

                    --last;
                }

                needsUpdate=true;

                break;
            }


            case core::componentmodel::topology::POINTSREMOVED:
            {
                //std::cout << "INFO_print : Col - POINTSREMOVED" << std::endl;
                if (tstc)
                {

                    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvsa=tstc->getTriangleVertexShellArray();
                    unsigned int last = tvsa.size() -1;

                    unsigned int i,j;
                    const sofa::helper::vector<unsigned int> tab = ( dynamic_cast< const sofa::component::topology::PointsRemoved * >( *itBegin ) )->getArray();

                    sofa::helper::vector<unsigned int> lastIndexVec;
                    for(unsigned int i_init = 0; i_init < tab.size(); ++i_init)
                    {

                        lastIndexVec.push_back(last - i_init);
                    }

                    for ( i = 0; i < tab.size(); ++i)
                    {
                        unsigned int i_next = i;
                        bool is_reached = false;
                        while( (!is_reached) && (i_next < lastIndexVec.size() - 1))
                        {

                            i_next += 1 ;
                            is_reached = is_reached || (lastIndexVec[i_next] == tab[i]);
                        }

                        if(is_reached)
                        {

                            lastIndexVec[i_next] = lastIndexVec[i];

                        }

                        const sofa::helper::vector<unsigned int> &shell=tvsa[lastIndexVec[i]];
                        for (j=0; j<shell.size(); ++j)
                        {

                            unsigned int ind_j =shell[j];

                            if ((unsigned)mytriangles[ind_j][0]==last)
                                mytriangles[ind_j][0]=tab[i];
                            else if ((unsigned)mytriangles[ind_j][1]==last)
                                mytriangles[ind_j][1]=tab[i];
                            else if ((unsigned)mytriangles[ind_j][2]==last)
                                mytriangles[ind_j][2]=tab[i];
                        }

                        if (debug_mode)
                        {

                            for (unsigned int j_loc=0; j_loc<mytriangles.size(); ++j_loc)
                            {

                                bool is_forgotten = false;
                                if ((unsigned)mytriangles[j_loc][0]==last)
                                {
                                    mytriangles[j_loc][0]=tab[i];
                                    is_forgotten=true;

                                }
                                else
                                {
                                    if ((unsigned)mytriangles[j_loc][1]==last)
                                    {
                                        mytriangles[j_loc][1]=tab[i];
                                        is_forgotten=true;

                                    }
                                    else
                                    {
                                        if ((unsigned)mytriangles[j_loc][2]==last)
                                        {
                                            mytriangles[j_loc][2]=tab[i];
                                            is_forgotten=true;
                                        }
                                    }

                                }

                                if(is_forgotten)
                                {

                                    unsigned int ind_forgotten = j;

                                    bool is_in_shell = false;
                                    for (unsigned int j_glob=0; j_glob<shell.size(); ++j_glob)
                                    {
                                        is_in_shell = is_in_shell || (shell[j_glob] == ind_forgotten);
                                    }

                                    if(!is_in_shell)
                                    {
                                        std::cout << "INFO_print : Col - triangle is forgotten in SHELL !!! global index = "  << ind_forgotten << std::endl;
                                    }

                                }

                            }
                        }

                        --last;
                    }

                }

                needsUpdate=true;

                break;
            }

            // Case "POINTSRENUMBERING" added to propagate the treatment to the Visual Model

            case core::componentmodel::topology::POINTSRENUMBERING:
            {
                //std::cout << "INFO_print : Vis - POINTSRENUMBERING" << std::endl;

                if (tstc)
                {

                    //const sofa::helper::vector<sofa::component::topology::Triangle> &ta=tstc->getTriangleArray();

                    unsigned int i;

                    const sofa::helper::vector<unsigned int> tab = ( dynamic_cast< const sofa::component::topology::PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

                    for ( i = 0; i < mytriangles.size(); ++i)
                    {
                        mytriangles[i][0]  = tab[mytriangles[i][0]];
                        mytriangles[i][1]  = tab[mytriangles[i][1]];
                        mytriangles[i][2]  = tab[mytriangles[i][2]];
                    }

                }

                //}

                break;

            }

            default:
                // Ignore events that are not Triangle  related.
                break;
            }; // switch( changeType )

            mytriangles.resize( elems.size() ); // not necessary
            resize( elems.size() ); // not necessary

            ++itBegin;
        } // while( changeIt != last; )
    }
    if (needsUpdate)
    {
        updateFlags();
    }
}

void TriangleModel::draw(int index)
{
    Triangle t(this,index);
    glBegin(GL_TRIANGLES);
    helper::gl::glNormalT(t.n());
    helper::gl::glVertexT(t.p1());
    helper::gl::glVertexT(t.p2());
    helper::gl::glVertexT(t.p3());
    glEnd();
}

void TriangleModel::draw()
{
    if (getContext()->getShowCollisionModels())
    {
        if (getContext()->getShowWireFrame())
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
            draw(i);
        }

        glColor3f(1.0f, 1.0f, 1.0f);
        glDisable(GL_LIGHTING);
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    if (getPrevious()!=NULL && getContext()->getShowBoundingCollisionModels())
        getPrevious()->draw();
}

void TriangleModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    updateFromTopology();
    if (needsUpdate && !cubeModel->empty()) cubeModel->resize(0);
    if (!isMoving() && !cubeModel->empty() && !needsUpdate) return; // No need to recompute BBox if immobile

    needsUpdate=false;
    Vector3 minElem, maxElem;
    const VecCoord& x = *this->mstate->getX();

    const bool calcNormals = computeNormals.getValue();

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
            if (calcNormals)
                for (int i=0; i<size; i++)
                {
                    Triangle t(this,i);
                    const Vector3& pt1 = x[t.p1Index()];
                    const Vector3& pt2 = x[t.p2Index()];
                    const Vector3& pt3 = x[t.p3Index()];

                    /*for (int c = 0; c < 3; c++)
                    {
                        if (i==0)
                        {
                    	minElem[c] = pt1[c];
                    	maxElem[c] = pt1[c];
                        }
                        else
                        {
                    	if (pt1[c] > maxElem[c]) maxElem[c] = pt1[c];
                    	else if (pt1[c] < minElem[c]) minElem[c] = pt1[c];
                        }
                        if (pt2[c] > maxElem[c]) maxElem[c] = pt2[c];
                        else if (pt2[c] < minElem[c]) minElem[c] = pt2[c];
                        if (pt3[c] > maxElem[c]) maxElem[c] = pt3[c];
                        else if (pt3[c] < minElem[c]) minElem[c] = pt3[c];
                    }*/

                    // Also recompute normal vector
                    t.n() = cross(pt2-pt1,pt3-pt1);
                    t.n().normalize();

                }
            cubeModel->setLeafCube(0, std::make_pair(this->begin(),this->end()), minElem, maxElem); // define the bounding box of the current triangle
        }
    }
    else
    {

        cubeModel->resize(size);  // size = number of triangles
        if (!empty())
        {
            for (int i=0; i<size; i++)
            {
                Triangle t(this,i);
                const Vector3& pt1 = x[t.p1Index()];
                const Vector3& pt2 = x[t.p2Index()];
                const Vector3& pt3 = x[t.p3Index()];

                for (int c = 0; c < 3; c++)
                {
                    minElem[c] = pt1[c];
                    maxElem[c] = pt1[c];
                    if (pt2[c] > maxElem[c]) maxElem[c] = pt2[c];
                    else if (pt2[c] < minElem[c]) minElem[c] = pt2[c];
                    if (pt3[c] > maxElem[c]) maxElem[c] = pt3[c];
                    else if (pt3[c] < minElem[c]) minElem[c] = pt3[c];
                }
                if (calcNormals)
                {
                    // Also recompute normal vector
                    t.n() = cross(pt2-pt1,pt3-pt1);
                    t.n().normalize();
                }
                cubeModel->setParentOf(i, minElem, maxElem); // define the bounding box of the current triangle
            }
            cubeModel->computeBoundingTree(maxDepth);
        }
    }
}

void TriangleModel::computeContinuousBoundingTree(double dt, int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    updateFromTopology();
    if (needsUpdate) cubeModel->resize(0);
    if (!isMoving() && !cubeModel->empty() && !needsUpdate) return; // No need to recompute BBox if immobile

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
    /*
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
    */
}

} // namespace collision

} // namespace component

} // namespace sofa

