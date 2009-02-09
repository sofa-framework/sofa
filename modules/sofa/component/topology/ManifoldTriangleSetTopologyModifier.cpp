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


} // namespace topology

} // namespace component

} // namespace sofa

