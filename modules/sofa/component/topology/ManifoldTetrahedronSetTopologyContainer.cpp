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


void ManifoldTetrahedronSetTopologyContainer::reinit()
{
    std::cout << "Starting functions tests: " << std::endl;
    //	int test;

    /*	Tetrahedron tetra_test=Tetrahedron(m_tetrahedron[1][0], m_tetrahedron[1][3], m_tetrahedron[1][2], m_tetrahedron[1][1]);
    test = getTetrahedronOrientation(m_tetrahedron[1], tetra_test);
    std::cout << "res: " << test<< std::endl;

    Tetrahedron tetra_test2=Tetrahedron(m_tetrahedron[1][2], m_tetrahedron[1][3], m_tetrahedron[1][0], m_tetrahedron[1][1]);
    test = getTetrahedronOrientation(m_tetrahedron[1], tetra_test2);
    std::cout << "res: " << test<< std::endl;
    */



    createTetrahedronEdgeShellArray();
    createTetrahedronTriangleShellArray();


    /*		for (unsigned int i = 0 ; i <m_edge.size();i++)
      std::cout << i  << " => " << m_edge[i] <<std::endl;

    for (unsigned int i = 0; i < m_tetrahedronEdgeShell.size(); i++)
      std::cout << i << " => " << m_tetrahedronEdgeShell[i] << std::endl;

    for (unsigned int i =0; i<m_tetrahedron.size();i++)
      std::cout << i << " => "<<m_tetrahedron[i] << std::endl;
    */
}


void ManifoldTetrahedronSetTopologyContainer::init()
{
    TetrahedronSetTopologyContainer::init();
}



void ManifoldTetrahedronSetTopologyContainer::createTetrahedronVertexShellArray ()
{
    std::cout << "ManifoldTetrahedronSetTopologyContainer::createTetrahedronVertexShellArray ()"<<std::endl;

    // TO be implemented
    // see late: for the topology, only one connexe composante around one vertex.

    TetrahedronSetTopologyContainer::createTetrahedronVertexShellArray();

}



void ManifoldTetrahedronSetTopologyContainer::createTetrahedronEdgeShellArray ()
{
    std::cout << "ManifoldTetrahedronSetTopologyContainer::createTetrahedronEdgeShellArray ()"<<std::endl;

    // Get edge array
    sofa::helper::vector<Edge> edges = getEdgeArray();

    // Creating Tetrahedrons edges shell unordered
    TetrahedronSetTopologyContainer::createTetrahedronEdgeShellArray();

    //	for (unsigned int i = 0; i < m_tetrahedronEdgeShell.size(); i++)
    //  std::cout << i << " => " << m_tetrahedronEdgeShell[i] << std::endl;


    for (unsigned int edgeIndex =0; edgeIndex<edges.size(); edgeIndex++)
    {

        sofa::helper::vector <unsigned int> &shell = getTetrahedronEdgeShellForModification (edgeIndex);
        sofa::helper::vector <unsigned int>::iterator it;
        sofa::helper::vector < sofa::helper::vector <unsigned int> > vertexTofind;
        sofa::helper::vector <unsigned int> goodShell;
        unsigned int firstVertex =0;
        unsigned int secondVertex =0;
        unsigned int cpt = 0;

        vertexTofind.resize (shell.size());

        // Path to follow creation
        for (unsigned int tetraIndex = 0; tetraIndex < shell.size(); tetraIndex++)
        {
            cpt = 0;

            for (unsigned int vertex = 0; vertex < 4; vertex++)
            {
                if(m_tetrahedron[shell[ tetraIndex]][vertex] != edges[edgeIndex][0] && m_tetrahedron[shell[ tetraIndex]][vertex] != edges[edgeIndex][1] )
                {
                    vertexTofind[tetraIndex].push_back (m_tetrahedron[shell[ tetraIndex]][vertex]);
                    cpt++;
                }

                if (cpt == 2)
                    break;
            }
        }

        Tetrahedron tetra_first = Tetrahedron(edges[edgeIndex][0], edges[edgeIndex][1], vertexTofind[0][0], vertexTofind[0][1]);

        int good = getTetrahedronOrientation (m_tetrahedron[shell[ 0]], tetra_first);

        if (good == 1) //then tetra is in good order, initialisation.
        {
            firstVertex = vertexTofind[0][0];
            secondVertex = vertexTofind[0][1];
        }
        else if (good == 0)
        {
            firstVertex = vertexTofind[0][1];
            secondVertex = vertexTofind[0][0];
        }
        else
        {
            std::cout << "Error: createTetrahedronEdgeShellArray: Houston there is a probleme." <<std::endl;
        }

        goodShell.push_back(shell[0]);

        bool testFind = false;
        bool reverse = false;
        cpt = 0;


        // Start following path
        for (unsigned int i = 1; i < shell.size(); i++)
        {
            for (unsigned int j = 1; j < shell.size(); j++)
            {

                Tetrahedron tetra_test1 = Tetrahedron(edges[edgeIndex][0], edges[edgeIndex][1], secondVertex, vertexTofind[j][1]);
                Tetrahedron tetra_test2 = Tetrahedron(edges[edgeIndex][0], edges[edgeIndex][1], secondVertex, vertexTofind[j][0]);

                if (vertexTofind[j][0] == secondVertex && getTetrahedronOrientation (m_tetrahedron[shell[ j]], tetra_test1) == 1) //find next tetra, in one or the other order.
                {
                    goodShell.push_back(shell[j]);
                    secondVertex = vertexTofind[j][1];
                    testFind = true;
                    break;
                }
                else if(vertexTofind[j][1] == secondVertex && getTetrahedronOrientation (m_tetrahedron[shell[ j]], tetra_test2) == 1)
                {
                    goodShell.push_back(shell[j]);
                    secondVertex = vertexTofind[j][0];
                    testFind = true;
                    break;
                }
            }

            if (!testFind) //tetra has not be found, this mean we reach a border, we reverse the method
            {
                reverse = true;
                break;
            }

            cpt++;
            testFind =false;
        }


        // Reverse path following methode
        if(reverse)
        {
#ifndef NDEBUG
            std::cout << "Edge on border: "<< edgeIndex << std::endl;
#endif
            for (unsigned int i = cpt+1; i<shell.size(); i++)
            {
                for (unsigned int j = 0; j<shell.size(); j++)
                {

                    Tetrahedron tetra_test1 = Tetrahedron(edges[edgeIndex][0], edges[edgeIndex][1], vertexTofind[j][1], firstVertex);
                    Tetrahedron tetra_test2 = Tetrahedron(edges[edgeIndex][0], edges[edgeIndex][1], vertexTofind[j][0], firstVertex);

                    if (vertexTofind[j][0] == firstVertex && getTetrahedronOrientation (m_tetrahedron[shell[ j]], tetra_test1) == 1) //find next tetra, in one or the other order.
                    {
                        goodShell.insert (goodShell.begin(),shell[j]);
                        firstVertex = vertexTofind[j][1];
                        testFind = true;
                        break;
                    }
                    else if(vertexTofind[j][1] == firstVertex && getTetrahedronOrientation (m_tetrahedron[shell[ j]], tetra_test2) == 1)
                    {
                        goodShell.insert (goodShell.begin(),shell[j]);
                        firstVertex = vertexTofind[j][0];
                        testFind = true;
                        break;
                    }
                }
            }
        }

        shell = goodShell;
        goodShell.clear();
        vertexTofind.clear();
    }
}


void ManifoldTetrahedronSetTopologyContainer::createTetrahedronTriangleShellArray ()
{
    //std::cout << "ManifoldTetrahedronSetTopologyContainer::createTetrahedronTriangleShellArray ()"<<std::endl;
    // To be implemented
    // at most 2 tetrahedrons adjacent to one triangle.

    TetrahedronSetTopologyContainer::createTetrahedronTriangleShellArray();

    //	for (unsigned int i = 0; i <m_tetrahedronTriangleShell.size();i++)
    // std::cout << i << " old => " << m_tetrahedronTriangleShell[i] << std::endl;


    for (unsigned int triangleIndex = 0; triangleIndex < m_tetrahedronTriangleShell.size(); triangleIndex++)
    {
        sofa::helper::vector <unsigned int> &shell = getTetrahedronTriangleShellForModification (triangleIndex);

        if (shell.size() == 1)
        {
            //on fait rien pour le moment mais il faudrait verifier que le triangle est bien dans le bon sens:
#ifndef NDEBUG
            int test = getTriangleTetrahedronOrientation (m_tetrahedron[ shell[0] ], m_triangle[ triangleIndex ]);
            std::cout << "Border test: " << test << std::endl;
#endif
        }
        else if (shell.size() == 2)
        {
            int test = getTriangleTetrahedronOrientation (m_tetrahedron[ shell[0] ], m_triangle[ triangleIndex ]);
            if ( test == 0) // not in good order in first tetra of the shell, need to swap
            {
                unsigned int buffer = shell[0];
                shell[0] = shell[1];
                shell[1] = buffer;
            }
        }
        else
        {
            std::cout << " Error: createTetrahedronTriangleShellArray, manifold topology is not fullfil" << std::endl;
        }
    }

    //	for (unsigned int i = 0; i <m_tetrahedronTriangleShell.size();i++)
    //	  std::cout << i << " new => " << m_tetrahedronTriangleShell[i] << std::endl;

}


bool ManifoldTetrahedronSetTopologyContainer::checkTopology() const
{
    std::cout << "ManifoldTetrahedronSetTopologyContainer::checkTopology ()"<<std::endl;
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
    std::cout << "ManifoldTetrahedronSetTopologyContainer::clear ()"<<std::endl;
    //To be completed if necessary

    TetrahedronSetTopologyContainer::clear();
}


int ManifoldTetrahedronSetTopologyContainer::getTetrahedronOrientation (const Tetrahedron &t_ref, const Tetrahedron &t_test )
{
    //	std::cout << "ManifoldTetrahedronSetTopologyContainer::getTetrahedronOrientation ()"<<std::endl;
#ifndef NDEBUG
    std::cout << "Tetra de ref: " << t_ref << std::endl;
    std::cout << "Tetra a tester: " << t_test << std::endl;
#endif

    std::map<unsigned int, unsigned int> mapPosition;
    unsigned int positionsChange[4];
    std::map<unsigned int, unsigned int>::iterator it;

    unsigned int permutation=0;
    unsigned int buffer;

    for (unsigned int i = 0; i< 4; i++)
    {
        mapPosition[t_ref[i]] = i;
    }

    for (unsigned int i= 0; i <4; i++)
    {
        it = mapPosition.find (t_test[i]);
        if (it == mapPosition.end())
        {
#ifndef NDEBUG
            std::cout <<"Error: getTetrahedronOrientation: reference and testing tetrahedrons are not composed by the same vertices."<<std::endl;
#endif
            return -1;
        }
        positionsChange[(*it).second] = i;
    }

    for (unsigned int i = 0; i <4; i++)
    {
        if( positionsChange[i] != i)
        {
            for (unsigned int j= i; j<4; j++)
            {
                if(positionsChange[j]==i)
                {
                    buffer = positionsChange[i];
                    positionsChange[i] = positionsChange[j];
                    positionsChange[j] = buffer;
                    permutation++;
                    break;
                }
            }
        }
    }

    if( permutation%2 == 0)
        return 1;
    else
        return 0;
}

int ManifoldTetrahedronSetTopologyContainer::getTriangleTetrahedronOrientation (const Tetrahedron &t, const Triangle &tri )
{
    //std::cout << "ManifoldTetrahedronSetTopologyContainer::getTriangleTetrahedronOrientation ()"<<std::endl;
    //To be done in better way:

    std::map <unsigned int, unsigned int> mapPosition;
    std::map<unsigned int, unsigned int>::iterator it;
    unsigned int positionsChange[3];
    sofa::helper::vector <Triangle> positifs;



    for (unsigned int i = 0; i< 4; i++)
    {
        mapPosition[t[i]] = i;
    }

    for (unsigned int i = 0; i<3; i++)
    {
        it = mapPosition.find (tri[i]);
        if (it == mapPosition.end())
        {
#ifndef NDEBUG
            std::cout <<"Error: getTriangleTetrahedronOrientation: tetrahedrons and triangle are not composed by the same vertices."<<std::endl;
#endif
            return -1;
        }
        positionsChange[i] = (*it).second;
    }


    // a la barbare:

    Triangle triBuf;

    triBuf = Triangle (1,2,3);  positifs.push_back (triBuf);
    triBuf = Triangle (2,3,1);  positifs.push_back (triBuf);
    triBuf = Triangle (3,1,2);  positifs.push_back (triBuf);

    triBuf = Triangle (3,2,0);  positifs.push_back (triBuf);
    triBuf = Triangle (2,0,3);  positifs.push_back (triBuf);
    triBuf = Triangle (0,3,2);  positifs.push_back (triBuf);

    triBuf = Triangle (0,1,3);  positifs.push_back (triBuf);
    triBuf = Triangle (1,3,0);  positifs.push_back (triBuf);
    triBuf = Triangle (3,0,1);  positifs.push_back (triBuf);

    triBuf = Triangle (2,1,0);  positifs.push_back (triBuf);
    triBuf = Triangle (1,0,2);  positifs.push_back (triBuf);
    triBuf = Triangle (0,2,1);  positifs.push_back (triBuf);

    for (unsigned int j =0; j<positifs.size(); j++)
    {
        if ( positionsChange[0] == positifs[j][0] && positionsChange[1] == positifs[j][1] && positionsChange[2] == positifs[j][2])
        {
            return 1;
        }
    }

    return 0;
}


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


void ManifoldTetrahedronSetTopologyContainer::draw()
{

    TetrahedronSetTopologyContainer::draw();


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
        sofa::component::container::MechanicalObject<Vec3Types>* dofs;
        this->getContext()->get(dofs);

        // Creating dofs
        if ( !dofs )
        {

            cerr << "Hexa2TriangleTopologicalMapping::buildTriangleMesh(). Error: can't find the DOFs on the hexahedron topology." << endl;
            return;
        }
        sofa::component::container::MechanicalObject<Vec3Types>::VecCoord& coords = *dofs->getX();

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

        for(unsigned int edge = 0; edge< 50/*m_edge.size()*/; edge++)
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

        for (unsigned int i = 0; i < 50/*m_edge.size()*/; i++)
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

        for (unsigned int i = 0; i < 50/*m_edge.size()*/; i++)
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
            sofa::component::container::MechanicalObject<Vec3Types>* dofs;
            this->getContext()->get(dofs);

            // Creating dofs
            if ( !dofs )
            {

                cerr << "Hexa2TriangleTopologicalMapping::buildTriangleMesh(). Error: can't find the DOFs on the hexahedron topology." << endl;
                return;
            }
            sofa::component::container::MechanicalObject<Vec3Types>::VecCoord& coords = *dofs->getX();


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

