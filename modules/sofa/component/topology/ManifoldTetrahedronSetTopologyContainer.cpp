/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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

#include <sofa/component/topology/ManifoldTetrahedronSetTopologyContainer.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/component/container/MeshLoader.h>

#include <sofa/helper/system/gl.h>
#include <sofa/helper/gl/template.h>

#include <sofa/helper/gl/glText.inl>
#include <sofa/component/container/MechanicalObject.inl>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace std;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(ManidfoldTetrahedronSetTopologyContainer)
int ManifoldTetrahedronSetTopologyContainerClass = core::RegisterObject("Manifold Tetrahedron set topology container")
        .add< ManifoldTetrahedronSetTopologyContainer >()
        ;

const unsigned int tetrahedronEdgeArray[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};

ManifoldTetrahedronSetTopologyContainer::ManifoldTetrahedronSetTopologyContainer()
    : TetrahedronSetTopologyContainer()// draw to be restored
    //, d_tetrahedron(initDataPtr(&d_tetrahedron, &m_tetrahedron, "tetras", "List of tetrahedron indices"))
    //, _draw(initData(&_draw, false, "drawTetras","if true, draw the tetr46ahedrons in the topology"))
{
    debugViewIndices=this->initData(&debugViewIndices, (bool) false, "debugViewTriangleIndices", "Debug : view triangles indices");
    debugViewIndicesTetra=this->initData(&debugViewIndicesTetra, (bool) false, "debugViewTetraIndices", "Debug : view tetra indices");
    shellDisplay=this->initData(&shellDisplay, (bool) false, "debugViewShells", "Debug : view shells tetra");
}

ManifoldTetrahedronSetTopologyContainer::ManifoldTetrahedronSetTopologyContainer(const sofa::helper::vector< Tetrahedron >& tetrahedra )
    : TetrahedronSetTopologyContainer( tetrahedra)
    //, m_tetrahedron( tetrahedra )
    //, d_tetrahedron(initDataPtr(&d_tetrahedron, &m_tetrahedron, "tetras", "List of tetrahedron indices"))
{

}


void ManifoldTetrahedronSetTopologyContainer::init()
{
    TetrahedronSetTopologyContainer::init();
}


void ManifoldTetrahedronSetTopologyContainer::createTetrahedronVertexShellArray ()
{

    // TO be implemented
    // see late: for the topology, only one connexe composante around one vertex.

    TetrahedronSetTopologyContainer::createTetrahedronVertexShellArray();

}

void ManifoldTetrahedronSetTopologyContainer::createTetrahedronEdgeShellArray ()
{

    // To be implemented :
    /*
      Tetrahedraons have to be oriented around each edges.
      Same algo as in 2d:

      - take the edge, third point find the next point in good order
      - use function getTetrahedronOrientation
      - loop
      - when map is done, order shell.
    */

    TetrahedronSetTopologyContainer::createTetrahedronEdgeShellArray();

}

void ManifoldTetrahedronSetTopologyContainer::createTetrahedronTriangleShellArray ()
{
    // To be implemented
    // at most 2 tetrahedrons adjacent to one triangle.


    TetrahedronSetTopologyContainer::createTetrahedronTriangleShellArray();
}


bool ManifoldTetrahedronSetTopologyContainer::checkTopology() const
{
#ifndef NDEBUG
    bool ret = true;

    // To be implemented later later....

    return ret && TetrahedronSetTopologyContainer::checkTopology();
#else
    return true;
#endif
}

void ManifoldTetrahedronSetTopologyContainer::clear()
{
    //To be completed if necessary

    TetrahedronSetTopologyContainer::clear();
}


bool ManifoldTetrahedronSetTopologyContainer::getTetrahedronOrientation (const Tetrahedron &t, const Tetrahedron &t_test )
{
    //To be implemented
    /*

      First tetra is in one orientation. We know the 4 points
      we search the orientation of a second tetra.

      - First confirm it is the same 4 points
      - look how many permutation needed to fin the same tetra.
      - if nbr permuation is pair, same orientation

      => idea use 0 1 map to make bit a bit tests

     */
    //no warnings:
    (void) t;
    (void) t_test;

    return true;

}

int ManifoldTetrahedronSetTopologyContainer::getTriangleTetrahedronOrientation (const Tetrahedron &t, const Triangle &tri )
{
    //To be implemented

    /*

    - equivalent to TriangleEdgeShell [i]
    - first triangle of the tetrahedron should be in positive orientation
    - This first triangle is the one on the border if tetrahedron is on border.
    - return either negatif or positive orientation in the tetrahedron or -1 if error.

    => should be used in createTetrahedronTriangleShellArray



      for(TetraID i = 0; i < m_nbTetras; ++i)
    {
        const Tetra& t = m_topo->getTetra(i);
        const TetraTriangles& tFaces = m_topo->getTriangleTetraShell(i);
        for(int l = 0; l < 4; ++l)
        {
            int sign = 1;
            const Triangle& f = m_topo->getTriangle(tFaces[l]);

            int m = 0;
            while(t[m] == f[0] || t[m] == f[1] || t[m] == f[2])
                ++m;
            if(m%2 == 1)
                sign *= -1;

             int n = 0;
             while(f[0] != t[n])
                ++n;

            if((n+1)%4 == m && f[2] == t[(n+2)%4])
                sign *= -1;
            if((n+1)%4 != m && f[2] == t[(n+1)%4])
                sign *= -1;
        }
    }
    */

    //no warnings:
    (void) t;
    (void) tri;

    return 0;
}


void ManifoldTetrahedronSetTopologyContainer::draw()
{

    if (debugViewIndicesTetra.getValue())
    {
        static int  dof_list;
        static int  tetra_list;
        static int  edges_list;
        static int  tetraEdge_list;

        Vector3 position;
        string text;
        double scale = 0.0005;

        // *** DOFS ***
        sofa::component::MechanicalObject<Vec3Types>* dofs;
        this->getContext()->get(dofs);

        // Creating dofs
        if ( !dofs )
        {

            cerr << "Hexa2TriangleTopologicalMapping::buildTriangleMesh(). Error: can't find the DOFs on the hexahedron topology." << endl;
            return;
        }
        sofa::component::MechanicalObject<Vec3Types>::VecCoord& coords = *dofs->getX();

        glColor4f ( 1,1,1,0 );
        // Drawing dofs
        dof_list = glGenLists(1);
        glNewList(dof_list, GL_COMPILE);

        for (unsigned int i = 0; i< coords.size(); i++)
        {
            sofa::helper::gl::GlText::draw ( i, coords[i], scale );
        }

        glEndList();



        // Creating tetra
        if (!hasTetrahedronVertexShell())
        {
            std::cout << "creating TriangleVertexShellArray()" << std::endl;
            createTetrahedronVertexShellArray();
        }

        //recupere les coord bary de chaque tri -> code surement existant deja, mais pas le temps:
        sofa::helper::vector< sofa::helper::vector<double> > bary_coord;
        sofa::helper::vector< sofa::helper::vector<double> > bary_coord2;

        Tetrahedron triVertex;
        float bary_x, bary_y, bary_z;
        bary_coord.resize(m_tetrahedron.size());

        // Creating barycentrique coord
        for (unsigned int tri =0; tri< m_tetrahedron.size(); tri++)
        {
            triVertex = m_tetrahedron[tri];

            bary_x=0;
            bary_y=0;
            bary_z=0;

            for (unsigned int i=0; i<4; i++)
            {
                bary_x = bary_x + coords[ triVertex[i] ][0];
                bary_y = bary_y + coords[ triVertex[i] ][1];
                bary_z = bary_z + coords[ triVertex[i] ][2];
            }

            bary_coord[tri].push_back(bary_x/4);
            bary_coord[tri].push_back(bary_y/4);
            bary_coord[tri].push_back(bary_z/4);
        }


        // Creating list for index drawing


        tetra_list = glGenLists(1);
        glNewList(tetra_list, GL_COMPILE);
        scale = 0.0005;
        glColor4f ( 1,1,1,0 );
        // Drawing triangles index
        for (unsigned int tri =0; tri< m_tetrahedron.size(); tri++)
        {
            position[0]=bary_coord[tri][0];
            position[1]=bary_coord[tri][1];
            position[2]=bary_coord[tri][2];
            sofa::helper::gl::GlText::draw ( tri, position , scale );
        }

        glEndList();



        // Display edge composition:
        if (!hasEdges())
            createEdgeSetArray();


        Edge the_edge;
        bary_coord2.resize(m_edge.size());

        for(unsigned int edge = 0; edge< m_edge.size(); edge++)
        {
            the_edge = getEdgeArray()[edge];

            bary_x=0;
            bary_y=0;
            bary_z=0;

            for (unsigned int i = 0; i<2; i++)
            {
                bary_x = bary_x + coords[ the_edge[i] ][0];
                bary_y = bary_y + coords[ the_edge[i] ][1];
                bary_z = bary_z + coords[ the_edge[i] ][2];
            }

            bary_coord2[edge].push_back(bary_x/2);
            bary_coord2[edge].push_back(bary_y/2);
            bary_coord2[edge].push_back(bary_z/2);
        }


        edges_list = glGenLists(1);
        glNewList(edges_list, GL_COMPILE);
        glColor4f ( 1,1,0,1 );
        // Drawing edges index
        scale = 0.0002;
        for (unsigned int edge =0; edge< m_edge.size(); edge++)
        {
            position[0]=bary_coord2[edge][0];
            position[1]=bary_coord2[edge][1];
            position[2]=bary_coord2[edge][2];
            sofa::helper::gl::GlText::draw ( edge, position , scale );
        }
        glEndList();


        // Display tetraEdgeShell positions:
        if (!hasTetrahedronEdgeShell())
            createTetrahedronEdgeShellArray();

        tetraEdge_list = glGenLists(1);
        glNewList(tetraEdge_list, GL_COMPILE);
        glColor4f ( 1,0,0,1 );
        scale = 0.0002;

        for (unsigned int i = 0; i < m_edge.size(); i++)
        {
            for (unsigned int j =0; j< m_tetrahedronEdgeShell[i].size(); j++)
            {
                position[0] = ( bary_coord[m_tetrahedronEdgeShell[i][j]][0] + bary_coord2[i][0]*9)/10;
                position[1] = ( bary_coord[m_tetrahedronEdgeShell[i][j]][1] + bary_coord2[i][1]*9)/10;
                position[2] = ( bary_coord[m_tetrahedronEdgeShell[i][j]][2] + bary_coord2[i][2]*9)/10;

                sofa::helper::gl::GlText::draw ( j, position , scale );
            }
        }
        glEndList();



        glCallList (dof_list);
        //  glCallList (tetra_list);
        glCallList (edges_list);
        glCallList (tetraEdge_list);

    }



    if (shellDisplay.getValue())
    {

        // Display edge composition:
        if (!hasEdges())
            createEdgeSetArray();

        for (unsigned int i = 0; i < m_edge.size(); i++)
        {
            std::cout <<"Edge: " << i << " vertex: " << m_edge[i] << std::endl;
        }

        /*
        // Display tetraEdgeShell positions:
        if (!hasTetrahedronEdgeShell())
        createTetrahedronEdgeShellArray();

        for (unsigned int i = 0; i < m_edge.size(); i++)
        {
        std::cout <<"Edge: " << i << " Shell: ";
        for (unsigned int j = 0; j < m_tetrahedronEdgeShell[i].size();j++)
        {
        std::cout << m_tetrahedronEdgeShell[i][j] << " ";
        }

        std::cout << std::endl;
        }*/
    }






    if (debugViewIndices.getValue())
    {
        static bool indexes_dirty = true;


        static int  dof_list;
        static int  triangles_list;
        static int  edges_list;
        //      static int  trishell_list;
        //      static int  edgeshell_list;

        indexes_dirty=true;

        if(indexes_dirty)
        {
            //	  std::cout << " passe la! " << std::endl;

            Vector3 position;
            string text;
            double scale = 0.0001;

            // *** DOFS ***
            sofa::component::MechanicalObject<Vec3Types>* dofs;
            this->getContext()->get(dofs);

            // Creating dofs
            if ( !dofs )
            {

                cerr << "Hexa2TriangleTopologicalMapping::buildTriangleMesh(). Error: can't find the DOFs on the hexahedron topology." << endl;
                return;
            }
            sofa::component::MechanicalObject<Vec3Types>::VecCoord& coords = *dofs->getX();


            // Drawing dofs
            dof_list = glGenLists(1);
            glNewList(dof_list, GL_COMPILE);

            for (unsigned int i = 0; i< coords.size(); i++)
            {
                sofa::helper::gl::GlText::draw ( i, coords[i], scale );
            }

            glEndList();


            //	const unsigned int nbrVertices = getNbPoints();
            const unsigned int nbrTriangles = getNumberOfTriangles();
            const unsigned int nbrEdges = getNumberOfEdges();

            // Creating triangles
            if (!hasTriangleVertexShell())
            {
                std::cout << "creating TriangleVertexShellArray()" << std::endl;
                createTriangleVertexShellArray();
            }


            //recupere les coord bary de chaque tri -> code surement existant deja, mais pas le temps:
            sofa::helper::vector< sofa::helper::vector<double> > bary_coord;


            Triangle triVertex;
            float bary_x, bary_y, bary_z;
            bary_coord.resize(nbrTriangles);

            // Creating barycentrique coord
            for (unsigned int tri =0; tri< nbrTriangles; tri++)
            {
                triVertex = getTriangleArray()[tri];

                bary_x=0;
                bary_y=0;
                bary_z=0;

                for (unsigned int i=0; i<3; i++)
                {
                    bary_x = bary_x + coords[ triVertex[i] ][0];
                    bary_y = bary_y + coords[ triVertex[i] ][1];
                    bary_z = bary_z + coords[ triVertex[i] ][2];
                }

                bary_coord[tri].push_back(bary_x/3);
                bary_coord[tri].push_back(bary_y/3);
                bary_coord[tri].push_back(bary_z/3);
            }


            // Creating list for index drawing


            triangles_list = glGenLists(1);
            glNewList(triangles_list, GL_COMPILE);
            scale = 0.00005;

            // Drawing triangles index
            for (unsigned int tri =0; tri< nbrTriangles; tri++)
            {
                position[0]=bary_coord[tri][0];
                position[1]=bary_coord[tri][1];
                position[2]=bary_coord[tri][2];
                sofa::helper::gl::GlText::draw ( tri, position , scale );
            }

            glEndList();

            /*
              trishell_list = glGenLists(1);
              glNewList(trishell_list, GL_COMPILE);

              // Drawing triangleVertexSHell positions around each vertex
              scale = 0.0008;
              for (unsigned int vert = 0; vert <nbrVertices; vert++)
              {

              for (unsigned int tri = 0; tri < m_triangleVertexShell[vert].size(); tri++)
              {
              position[0] = (coords[ vert ][0] * 1.5 + bary_coord[ m_triangleVertexShell[vert][tri] ][0])/2.5;
              position[1] = (coords[ vert ][1] * 1.5 + bary_coord[ m_triangleVertexShell[vert][tri] ][1])/2.5;
              position[2] = (coords[ vert ][2] * 1.5 + bary_coord[ m_triangleVertexShell[vert][tri] ][2])/2.5;


              sofa::helper::gl::GlText::draw ( tri, position , scale );
              }
              }
            	  glEndList();


              // Creatring edges
              if (!hasEdgeVertexShell())
              {
              std::cout << "creating createEdgeVertexShellArray()" << std::endl;

              createEdgeVertexShellArray();
              }

            */
            bary_coord.clear();
            Edge the_edge;
            bary_coord.resize(nbrEdges);

            for(unsigned int edge = 0; edge< nbrEdges; edge++)
            {
                the_edge = getEdgeArray()[edge];

                bary_x=0;
                bary_y=0;
                bary_z=0;

                for (unsigned int i = 0; i<2; i++)
                {
                    bary_x = bary_x + coords[ the_edge[i] ][0];
                    bary_y = bary_y + coords[ the_edge[i] ][1];
                    bary_z = bary_z + coords[ the_edge[i] ][2];
                }

                bary_coord[edge].push_back(bary_x/2);
                bary_coord[edge].push_back(bary_y/2);
                bary_coord[edge].push_back(bary_z/2);
            }


            edges_list = glGenLists(1);
            glNewList(edges_list, GL_COMPILE);

            // Drawing edges index
            scale = 0.00005;
            for (unsigned int edge =0; edge< nbrEdges; edge++)
            {
                position[0]=bary_coord[edge][0];
                position[1]=bary_coord[edge][1];
                position[2]=bary_coord[edge][2];
                sofa::helper::gl::GlText::draw ( edge, position , scale );
            }
            glEndList();

            /*
              edgeshell_list = glGenLists(1);
              glNewList(edgeshell_list, GL_COMPILE);

              // Drawing edgeVertexSHell positions around each vertex
              scale = 0.0004;
              for (unsigned int vert = 0; vert <nbrVertices; vert++)
              {

              for (unsigned int edge = 0; edge < m_edgeVertexShell[vert].size(); edge++)
              {
              position[0] = (coords[ vert ][0] * 1.5 + bary_coord[ m_edgeVertexShell[vert][edge] ][0])/2.5;
              position[1] = (coords[ vert ][1] * 1.5 + bary_coord[ m_edgeVertexShell[vert][edge] ][1])/2.5;
              position[2] = (coords[ vert ][2] * 1.5 + bary_coord[ m_edgeVertexShell[vert][edge] ][2])/2.5;


              sofa::helper::gl::GlText::draw ( edge, position , scale );
              }
              }

              glEndList();
            */
            indexes_dirty = false;
        }

        glCallList (dof_list);
        glCallList (triangles_list);
        glCallList (edges_list);
        //glCallList (trishell_list);
        //glCallList (edgeshell_list);
    }

}



} // namespace topology

} // namespace component

} // namespace sofa

