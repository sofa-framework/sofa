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
#include <sofa/component/topology/ManifoldTriangleSetTopologyModifier.h>
//#include <sofa/component/topology/TriangleSetTopologyChange.h>
#include <sofa/component/topology/ManifoldTriangleSetTopologyContainer.h>
#include <sofa/component/topology/TriangleSetTopologyContainer.h>
#include <algorithm>
//#include <functional>
#include <iostream>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{
SOFA_DECL_CLASS(ManifoldTriangleSetTopologyModifier)
int ManifoldTriangleSetTopologyModifierClass = core::RegisterObject("Triangle set topology manifold modifier")
        .add< ManifoldTriangleSetTopologyModifier >()
        ;

using namespace std;
using namespace sofa::defaulttype;


void ManifoldTriangleSetTopologyModifier::init()
{
    TriangleSetTopologyModifier::init();
    this->getContext()->get(m_container);
}



bool ManifoldTriangleSetTopologyModifier::removePrecondition(sofa::helper::vector< unsigned int >& items)
{
    createFutureModifications(items);
    createFutureModificationsEdge (items);

    return testRemoveModifications();
}



void ManifoldTriangleSetTopologyModifier::createFutureModifications(sofa::helper::vector< unsigned int >& items)
{
    Triangle vertexTriangle;
    sofa::helper::vector<unsigned int> triangleVertexShell;

    // Loop if there are many triangles to remove in the vector items
    for (unsigned int triangleIndex = 0; triangleIndex < items.size(); triangleIndex++)
    {
        vertexTriangle = m_container->getTriangleArray()[items[triangleIndex]];

        // Loop on the vertex composing the triangle
        for (unsigned int vertexIndex = 0; vertexIndex < 3; vertexIndex++)
        {

            triangleVertexShell = m_container->getTriangleVertexShellForModification(vertexTriangle[vertexIndex]);

            //search in the map of modification the index of the current triangle to remove
            it_modif = m_modifications.find(vertexTriangle[vertexIndex]);

            //If not found, insert a new line in the map: key = index of triangle
            //values: vector equivalent to triangleVertexShell with 0 for triangles to keep and 1 for triangle to remove
            if (it_modif == m_modifications.end())
            {
                m_modifications[vertexTriangle[vertexIndex]]=sofa::helper::vector<unsigned int>();

                for (unsigned int i = 0; i < triangleVertexShell.size(); i++)
                {
                    if(triangleVertexShell[i]==items[triangleIndex])
                    {
                        m_modifications[vertexTriangle[vertexIndex]].push_back(1);
                    }
                    else
                    {
                        m_modifications[vertexTriangle[vertexIndex]].push_back(0);
                    }
                }
            }
            else //If already exist, just change the value, of triangle to remove in the map: 0 => 1
            {
                for (unsigned int i = 0; i < triangleVertexShell.size(); i++)
                {
                    if(triangleVertexShell[i]==items[triangleIndex])
                    {
                        m_modifications[vertexTriangle[vertexIndex]][i]=1;
                    }
                }
            }
        }
    }
}



bool ManifoldTriangleSetTopologyModifier::testRemoveModifications()
{
    std::map< unsigned int, sofa::helper::vector<unsigned int> >::iterator it;
    sofa::helper::vector <PointID> border = m_container->getPointsBorder();

    unsigned int connexite;
    bool bord;
    bool test=true;


    for(it=m_modifications.begin(); it !=m_modifications.end(); it++)
    {

        bord=false;

        //Test border
        for (unsigned int i = 0; i<border.size(); i++)
        {
            if (border[i] == (*it).first)
            {
                m_modifications[(*it).first].push_back(1);
                bord=true;
            }
        }


        connexite = 0;
        for (unsigned int i = 0; i < ((*it).second).size()-1; i++)
        {

            if( ((*it).second)[i] != ((*it).second)[i+1] )
            {
                connexite++;
            }
        }

        //End the loop
        if( ((*it).second)[0] != ((*it).second)[((*it).second).size()-1] )
        {
            connexite++;
        }

        if (bord)
        {
            m_modifications[(*it).first].pop_back();
        }

        if( connexite > 2)
        {
            std::cout << "Error: ManifoldTriangleSetTopologyModifier::testRemoveModifications: You could not remove this/these triangle(s)";
            std::cout << " around the vertex: " << (*it).first << std::endl;

            test=false;
        }
    }


    if( test == false )
    {
        if (!m_modifications.empty())
            m_modifications.clear();

        if (!m_modificationsEdge.empty())
            m_modificationsEdge.clear();
    }

    return test;
}



void ManifoldTriangleSetTopologyModifier::createFutureModificationsEdge (const sofa::helper::vector <unsigned int> items)
{
    TriangleEdges TriangleEdgeArray;
    bool test = true;

    for (unsigned int  i = 0; i < items.size(); i++)
    {
        TriangleEdgeArray = m_container->getTriangleEdge( items[i] );

        for (unsigned int j =0; j < 3 ; j++)
        {

            for (unsigned int k =0; k< m_modificationsEdge.size(); k++)
            {
                if (TriangleEdgeArray[j] == m_modificationsEdge[k])
                {
                    test = false;
                    break;
                }
            }

            if (test)
            {
                m_modificationsEdge.push_back(TriangleEdgeArray[j]);
            }
            test = true;
        }
    }
}


void ManifoldTriangleSetTopologyModifier::removePostProcessing(const sofa::helper::vector< unsigned int >& edgeToBeRemoved, const sofa::helper::vector< unsigned int >& vertexToBeRemoved )
{
    removePostProcessingEdges();
    removePostProcessingTriangles();

    updateModifications( edgeToBeRemoved, vertexToBeRemoved);

    reorderEdgeForRemoving();
}



void ManifoldTriangleSetTopologyModifier::removePostProcessingEdges()
{

    sofa::helper::vector<unsigned int> vertexshell;
    bool test = false;

    for(it_modif=m_modifications.begin(); it_modif !=m_modifications.end(); ++it_modif)
    {

        for (unsigned int i=0; i<((*it_modif).second).size(); i++)
        {
            if( ((*it_modif).second)[i] == 1 )
            {
                test = true;
            }

            if ( ((*it_modif).second)[i] == 0 && test == true )
            {

                vertexshell=m_container->getEdgeVertexShellForModification((*it_modif).first);

                for (unsigned int j = 0; j<i; j++)
                {
                    vertexshell.push_back (vertexshell.front() );
                    vertexshell.erase ( vertexshell.begin() );
                }

                m_container->getEdgeVertexShellForModification((*it_modif).first) = vertexshell;

                break;
            }
        }

        test = false;
    }
}



void ManifoldTriangleSetTopologyModifier::removePostProcessingTriangles()
{
    std::map< unsigned int, sofa::helper::vector<unsigned int> >::iterator it;
    sofa::helper::vector<unsigned int> vertexshell;

    for(it=m_modifications.begin(); it !=m_modifications.end(); ++it)
    {

        for (unsigned int i=0; i<((*it).second).size(); i++)
        {
            if( ((*it).second)[i] == 1 )
            {
                vertexshell=m_container->getTriangleVertexShellForModification((*it).first);

                for (unsigned int j = 0; j<i; j++)
                {
                    vertexshell.push_back (vertexshell.front() );
                    vertexshell.erase ( vertexshell.begin() );
                }

                m_container->getTriangleVertexShellForModification((*it).first) = vertexshell;

                break;
            }
        }
    }

    if (!m_modifications.empty())
        m_modifications.clear();
}


void ManifoldTriangleSetTopologyModifier::reorderEdgeForRemoving()
{
    for (unsigned int i = 0; i < m_modificationsEdge.size(); i++)
    {
        m_container->reorderingEdge( m_modificationsEdge[i] );

    }

    m_modificationsEdge.clear();
}



void ManifoldTriangleSetTopologyModifier::updateModifications (const sofa::helper::vector< unsigned int >& edgeToBeRemoved,
        const sofa::helper::vector< unsigned int >& vertexToBeRemoved)
{
    //???
    for (unsigned int i = 0; i <vertexToBeRemoved.size(); i++)
    {
        it_modif = m_modifications.find( vertexToBeRemoved[i] );

        if(it_modif != m_modifications.end())
            m_modifications.erase( vertexToBeRemoved[i] );
    }


    for (unsigned int i = 0; i <edgeToBeRemoved.size(); i++)
    {
        for (unsigned int j = 0; j<m_modificationsEdge.size(); j++)
        {
            if (m_modificationsEdge[j] == edgeToBeRemoved[i])
            {
                m_modificationsEdge.erase(m_modificationsEdge.begin()+j);
                break;
            }
        }
    }

}


void ManifoldTriangleSetTopologyModifier::Debug() //To be removed when release is sure
{
#ifndef NDEBUG

    std::cout << "ManifoldTriangleSetTopologyModifier::Debug()" << std::endl;

    for (unsigned int i = 0; i < m_container->getNbPoints(); i++)
    {
        std::cout << "vertex: " << i << " => Triangles:  " << m_container->getTriangleVertexShellForModification(i) << std::endl;
    }

    for (unsigned int i = 0; i < m_container->getNumberOfEdges(); i++)
    {
        std::cout << "edge: " << i << " => Triangles:  " << m_container->getTriangleEdgeShellForModification(i) << std::endl;
    }

    for (unsigned int i = 0; i < m_container->getNumberOfEdges(); i++)
    {
        //	  std::cout << "vertex: " << i << " => Edges:  " << m_container->getEdgeVertexShellForModification(i) << std::endl;
    }
#endif
}



bool ManifoldTriangleSetTopologyModifier::addPrecondition( const sofa::helper::vector <Triangle> &triangles)
{

    std::map< unsigned int, sofa::helper::vector <Triangle> > extremes;
    std::map< unsigned int, Triangle > trianglesList;
    std::map< unsigned int, Triangle >::iterator it;
    sofa::helper::vector <unsigned int> listDone;

    unsigned int position[3];

    bool allDone = true;
    bool oneDone = true;

    // Copy the triangles vector with this positions as key:
    for (unsigned int i = 0; i < triangles.size(); i++)
    {
        trianglesList.insert ( pair <unsigned int, Triangle> (i, triangles[i]));
    }


    while ( trianglesList.size() != 0 || allDone == true)
    {
        //initialisation
        allDone = false;

        // horrible loop
        for ( it = trianglesList.begin(); it != trianglesList.end(); it++)
        {
            oneDone = true;

            for (unsigned int vertexIndex = 0; vertexIndex <3; vertexIndex++)
            {

                it_add = m_Addmodifications.find( (*it).second[vertexIndex]);

                //Fill map of extremes triangles and map m_addmodifications:
                if (it_add == m_Addmodifications.end())
                {
                    extremes[ (*it).second[vertexIndex] ] = sofa::helper::vector <Triangle> ();
                    m_Addmodifications[ (*it).second[vertexIndex] ] = sofa::helper::vector <int> ();

                    sofa::helper::vector <unsigned int> &triangleVertexShell = m_container->getTriangleVertexShellForModification((*it).second[vertexIndex]);

                    extremes[(*it).second[vertexIndex]].push_back( m_container->getTriangleArray()[triangleVertexShell[0]] );
                    extremes[(*it).second[vertexIndex]].push_back( m_container->getTriangleArray()[triangleVertexShell[ triangleVertexShell.size()-1 ]] );

                    for (unsigned int i=0; i<triangleVertexShell.size(); i++)
                    {
                        m_Addmodifications[ (*it).second[vertexIndex] ].push_back(-1);
                    }
                }


                //Tests where the triangle could be in the shell: i.e to which extreme triangle it is adjacent ATTENTION extrems could not be similar to shell
                if ( (*it).second[ (vertexIndex+1)%3 ] == extremes[(*it).second[vertexIndex]][1][(vertexIndex+2)%3] )
                {
                    //Should be added to the end of the shell
                    position[vertexIndex] = 1;
                }
                else if ( (*it).second[ (vertexIndex+2)%3 ] == extremes[(*it).second[vertexIndex]][0][(vertexIndex+1)%3] )
                {
                    //Should be added to the begining of the shell
                    position[vertexIndex] = 0;
                }
                else
                {
                    oneDone = false;
                    std::cout << " Error: ManifoldTriangleSetTopologyModifier::addPrecondition adding this triangle: ";
                    std::cout << (*it).second << " doesn't keep the topology manifold." << std::endl;
                }
            }

            if (oneDone)
                //really fill m_addmodifications and update extremes map for further triangles
                for (unsigned int vertexIndex = 0; vertexIndex <3; vertexIndex++)
                {
                    extremes[ (*it).second[vertexIndex] ][ position[vertexIndex] ] = (*it).second;
                    allDone = true;

                    if ( position[vertexIndex] == 0)
                    {
                        sofa::helper::vector <int>::iterator it_vec;
                        it_vec = m_Addmodifications[ (*it).second[vertexIndex] ].begin();
                        m_Addmodifications[ (*it).second[vertexIndex] ].insert( it_vec, (int)(*it).first);
                    }
                    else if ( position[vertexIndex] == 1)
                    {
                        m_Addmodifications[ (*it).second[vertexIndex] ].push_back( (int)(*it).first );
                    }

                    listDone.push_back((*it).first); //we can't erase it while in the same loop
                }
        }

        for(unsigned int i = 0; i < listDone.size(); i++)
        {
            trianglesList.erase( listDone[i] );
        }
        listDone.clear();

        // 4 case possible:
        // - allDone = false, list.size() != 0 => never add something in the for loop, couldn't go further. leave while.
        // - allDone = true, list.zie != 0 => not all done but they were some modification, it could change in next loop for
        // - allDone = false, list.size() == 0=> normaly not possible.
        // - allDone = true , list.size() == 0 => They all have been added. Congreatulation, leave while.
    }

    if (trianglesList.size() != 0 )
    {
        m_Addmodifications.clear();
        return false;
    }



    // For manifold classes all shells have to be present while adding triangles. As it is not obliged in upper class. It is done here.
    if(!m_container->hasTriangleVertexShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyModifier::addPrecondition] Triangle vertex shell array is empty." << std::endl;
#endif
        m_container->createTriangleVertexShellArray();
    }

    if(!m_container->hasTriangleEdgeShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyModifier::addPrecondition] Triangle edge shell array is empty." << std::endl;
#endif
        m_container->createTriangleEdgeShellArray();
    }

    if(!m_container->hasEdgeVertexShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyModifier::addPrecondition] Edge vertex shell array is empty." << std::endl;
#endif
        m_container->createEdgeVertexShellArray();
    }


    return true;
}



void ManifoldTriangleSetTopologyModifier::addPostProcessing(const sofa::helper::vector <Triangle> &triangles)
{

    // for each vertex, reorder shells:
    for (it_add = m_Addmodifications.begin(); it_add != m_Addmodifications.end(); it_add++)
    {
        sofa::helper::vector <unsigned int> &triangleVertexShell = m_container->getTriangleVertexShellForModification((*it_add).first);
        sofa::helper::vector <unsigned int> &edgeVertexShell = m_container->getEdgeVertexShellForModification((*it_add).first);

        sofa::helper::vector <unsigned int> triShellTmp = triangleVertexShell;
        sofa::helper::vector <unsigned int> edgeShellTmp = edgeVertexShell;

        bool before = true;
        unsigned int bord = 0;
        unsigned int bord2 = 0;

        if ( triShellTmp.size() != ((*it_add).second).size())
            std::cout << " Error: ManifoldTriangleSetTopologyModifier::addPostProcessing: Size of shells differ. " << std::endl;

        //reordering m_triangleVertexShell
        for (unsigned int i = 0; i <triShellTmp.size(); i++)
        {

            if ( (*it_add).second[i] == -1)
            {
                triShellTmp[i] = triangleVertexShell[i];
            }
            else
            {
                const Triangle &tri = triangles[(*it_add).second[i]];

                int indexTriangle = m_container->getTriangleIndex(tri[0],tri[1],tri[2]);

                triShellTmp[i] = indexTriangle;
            }
        }

        triangleVertexShell = triShellTmp;

        for (unsigned int i = 0; i <((*it_add).second).size(); i++)
        {

            if ( (*it_add).second[i] == -1)
            {
                edgeShellTmp[i] = edgeVertexShell[i];
                before = false;
                bord2=i;
                bord++;
            }
            else
            {
                const Triangle &tri = triangles[(*it_add).second[i]];

                int vertexInTriangle = m_container->getVertexIndexInTriangle(tri, (*it_add).first);

                if (before)
                {
                    if ( tri[ vertexInTriangle ] < tri[ (vertexInTriangle+1)%3 ] )
                        edgeShellTmp[i] = m_container->getEdgeIndex(tri[ vertexInTriangle ], tri[ (vertexInTriangle+1)%3 ]);
                    else
                        edgeShellTmp[i] = m_container->getEdgeIndex(tri[ (vertexInTriangle+1)%3 ], tri[ vertexInTriangle ]);

                    //m_triangleEdgeShell:
                    sofa::helper::vector <unsigned int> &triangleEdgeShell = m_container->getTriangleEdgeShellForModification(edgeShellTmp[i]);
                    unsigned int tmp = triangleEdgeShell[0];
                    triangleEdgeShell[0] = triangleEdgeShell[1];
                    triangleEdgeShell[1] = tmp;
                }
                else
                {
                    if ( tri[ vertexInTriangle ] < tri[ (vertexInTriangle+2)%3 ] )
                        edgeShellTmp[i] = m_container->getEdgeIndex(tri[ vertexInTriangle ], tri[ (vertexInTriangle+2)%3 ]);
                    else
                        edgeShellTmp[i] = m_container->getEdgeIndex(tri[ (vertexInTriangle+2)%3 ], tri[ vertexInTriangle ]);
                }
            }
        }

        if ( ((*it_add).second).size() != edgeShellTmp.size()) // we are on the border, one edge is missing
        {
            sofa::helper::vector <unsigned int>::iterator it;
            it = edgeShellTmp.begin();
            edgeShellTmp.insert (it+bord2+1, edgeVertexShell[bord+1]);
        }

        edgeVertexShell = edgeShellTmp;

    }

}



} // namespace topology

} // namespace component

} // namespace sofa

