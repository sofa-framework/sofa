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
#ifndef SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGY_INL
#define SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGY_INL

#include <sofa/component/topology/TetrahedronSetTopology.h>
#include <sofa/component/topology/PointSetTopology.inl>
#include <sofa/component/topology/TopologyChangedEvent.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/simulation/tree/GNode.h>
#include <algorithm>
#include <functional>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;


/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////TetrahedronSetTopology///////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class DataTypes>
TetrahedronSetTopology<DataTypes>::TetrahedronSetTopology(MechanicalObject<DataTypes> *obj)
    : TriangleSetTopology<DataTypes>( obj)
{
}

template<class DataTypes>
void TetrahedronSetTopology<DataTypes>::createComponents()
{
    this->m_topologyContainer = new TetrahedronSetTopologyContainer(this);
    this->m_topologyModifier= new TetrahedronSetTopologyModifier<DataTypes>(this);
    this->m_topologyAlgorithms= new TetrahedronSetTopologyAlgorithms<DataTypes>(this);
    this->m_geometryAlgorithms= new TetrahedronSetGeometryAlgorithms<DataTypes>(this);
}

template<class DataTypes>
void TetrahedronSetTopology<DataTypes>::init()
{
    TriangleSetTopology<DataTypes>::init();
}


/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////TetrahedronSetTopologyModifier//////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////


const unsigned int tetrahedronEdgeArray[6][2]= {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

template<class DataTypes>
class TetrahedronSetTopologyLoader : public PointSetTopologyLoader<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;

    VecCoord pointArray;
    TetrahedronSetTopologyModifier<DataTypes> *tstm;

    TetrahedronSetTopologyLoader(TetrahedronSetTopologyModifier<DataTypes> *tm) :PointSetTopologyLoader<DataTypes>(), tstm(tm)
    {
    }

    virtual void addTetra(int p1, int p2, int p3,int p4)
    {
        tstm->addTetrahedron(Tetrahedron(helper::make_array<unsigned int>((unsigned int)p1,(unsigned int)p2,(unsigned int) p3,(unsigned int) p4)));
    }
};
template<class DataTypes>
void TetrahedronSetTopologyModifier<DataTypes>::addTetrahedron(Tetrahedron t)
{

    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>( topology->getTopologyContainer() );
    assert (container != 0);
    container->m_tetrahedron.push_back(t);

}
template<class DataTypes>
bool TetrahedronSetTopologyModifier<DataTypes>::load(const char *filename)
{

    TetrahedronSetTopologyLoader<DataTypes> loader(this);
    if (!loader.load(filename))
        return false;
    else
    {
        loadPointSet(&loader);

        return true;
    }
}

template<class DataTypes>
void TetrahedronSetTopologyModifier<DataTypes>::writeMSHfile(const char *filename)
{

    std::ofstream myfile;
    myfile.open (filename);

    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);

    PointSetTopology< Vec3Types >* psp = dynamic_cast< PointSetTopology< Vec3Types >* >( topology );
    PointSetTopologyContainer * c_psp = static_cast< PointSetTopologyContainer* >(psp->getTopologyContainer());

    Vec3Types::VecCoord &p = *psp->getDOF()->getX();

    myfile << "$NOD\n";
    myfile << c_psp->getNumberOfVertices() <<"\n";

    for (unsigned int i=0; i<c_psp->getNumberOfVertices(); ++i)
    {

        double x = (double) p[i][0];
        double y = (double) p[i][1];
        double z = (double) p[i][2];

        myfile << i+1 << " " << x << " " << y << " " << z <<"\n";
    }

    myfile << "$ENDNOD\n";
    myfile << "$ELM\n";

    TetrahedronSetTopologyContainer * container = static_cast< TetrahedronSetTopologyContainer* >(topology->getTopologyContainer());
    const sofa::helper::vector<Tetrahedron> &tea=container->getTetrahedronArray();

    myfile << tea.size() <<"\n";

    for (unsigned int i=0; i<tea.size(); ++i)
    {
        myfile << i+1 << " 4 1 1 4 " << tea[i][0]+1 << " " << tea[i][1]+1 << " " << tea[i][2]+1 << " " << tea[i][3]+1 <<"\n";
    }

    myfile << "$ENDELM\n";

    myfile.close();

}

template<class DataTypes>
void TetrahedronSetTopologyModifier<DataTypes>::addTetrahedraProcess(const sofa::helper::vector< Tetrahedron > &tetrahedra)
{
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>( topology->getTopologyContainer() );
    assert (container != 0);
    if (container->m_tetrahedron.size()>0)
    {
        unsigned int tetrahedronIndex;
        const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvsa=container->getTetrahedronVertexShellArray();
        const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tesa=container->getTetrahedronEdgeShellArray();
        const sofa::helper::vector< sofa::helper::vector<unsigned int> > &ttsa=container->getTetrahedronTriangleShellArray();

        unsigned int j;


        for (unsigned int i = 0; i < tetrahedra.size(); ++i)
        {
            const Tetrahedron &t = tetrahedra[i];
            // check if the 3 vertices are different
            assert(t[0]!=t[1]);
            assert(t[0]!=t[2]);
            assert(t[0]!=t[3]);
            assert(t[1]!=t[2]);
            assert(t[1]!=t[3]);
            assert(t[2]!=t[3]);
            // check if there already exists a tetrahedron with the same indices
            assert(container->getTetrahedronIndex(t[0],t[1],t[2],t[3])== -1);
            container->m_tetrahedron.push_back(t);
            tetrahedronIndex=container->m_tetrahedron.size() - 1 ;

            if (tvsa.size()>0)
            {
                for (j=0; j<4; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell = container->getTetrahedronVertexShellForModification( t[j] );
                    shell.push_back( tetrahedronIndex );
                    sort(shell.begin(), shell.end());
                }
            }

            if (container->m_tetrahedronTriangle.size()>0)
            {
                int triangleIndex;
                for (j=0; j<4; ++j)
                {
                    triangleIndex=container->getTriangleIndex(t[(j+1)%4],t[(j+2)%4],
                            t[(j+3)%4]);
                    //assert(triangleIndex!= -1);

                    if(triangleIndex == -1)
                    {

                        // first create the traingle
                        sofa::helper::vector< Triangle > v;
                        Triangle e1 (t[(j+1)%4],t[(j+2)%4],t[(j+3)%4]);
                        v.push_back(e1);

                        addTrianglesProcess((const sofa::helper::vector< Triangle > &) v);

                        triangleIndex=container->getTriangleIndex(t[(j+1)%4],t[(j+2)%4],t[(j+3)%4]);
                        sofa::helper::vector< unsigned int > triangleIndexList;
                        triangleIndexList.push_back(triangleIndex);
                        this->addTrianglesWarning( v.size(), v,triangleIndexList);
                    }

                    container->m_tetrahedronTriangle.resize(triangleIndex+1);
                    container->m_tetrahedronTriangle[tetrahedronIndex][j]= triangleIndex;
                }
            }

            if (container->m_tetrahedronEdge.size()>0)
            {
                int edgeIndex;
                for (j=0; j<6; ++j)
                {
                    edgeIndex=container->getEdgeIndex(tetrahedronEdgeArray[j][0],
                            tetrahedronEdgeArray[j][1]);
                    assert(edgeIndex!= -1);

                    container->m_tetrahedronEdge.resize(edgeIndex+1);
                    container->m_tetrahedronEdge[tetrahedronIndex][j]= edgeIndex;
                }
            }

            if (tesa.size()>0)
            {
                for (j=0; j<6; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell = container->m_tetrahedronEdgeShell[container->m_tetrahedronEdge[tetrahedronIndex][j]];
                    shell.push_back( tetrahedronIndex );
                    sort(shell.begin(), shell.end());
                }
            }
            if (ttsa.size()>0)
            {
                for (j=0; j<4; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell = container->m_tetrahedronTriangleShell[container->m_tetrahedronTriangle[tetrahedronIndex][j]];
                    shell.push_back( tetrahedronIndex );
                    sort(shell.begin(), shell.end());
                }
            }
        }
    }
}



template<class DataTypes>
void TetrahedronSetTopologyModifier<DataTypes>::addTetrahedraWarning(const unsigned int nTetrahedra, const sofa::helper::vector< Tetrahedron >& tetrahedraList,
        const sofa::helper::vector< unsigned int >& tetrahedraIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    // Warning that tetrahedra just got created
    TetrahedraAdded *e=new TetrahedraAdded(nTetrahedra, tetrahedraList,tetrahedraIndexList,ancestors,baryCoefs);
    this->addTopologyChange(e);
}




template<class DataTypes>
void TetrahedronSetTopologyModifier<DataTypes>::removeTetrahedraWarning( sofa::helper::vector<unsigned int> &tetrahedra )
{
    /// sort vertices to remove in a descendent order
    std::sort( tetrahedra.begin(), tetrahedra.end(), std::greater<unsigned int>() );

    // Warning that these edges will be deleted
    TetrahedraRemoved *e=new TetrahedraRemoved(tetrahedra);
    this->addTopologyChange(e);
}



template<class DataTypes>
void TetrahedronSetTopologyModifier<DataTypes>::removeTetrahedraProcess( const sofa::helper::vector<unsigned int> &indices,const bool removeIsolatedItems)
{
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);


    /// only remove isolated edges if the structures exists since removeEdges
    /// will remove isolated vertices
    //if (removeIsolatedItems)
    //{
    /// force the creation of the Tetrahedron Edge Shell array to detect isolated edges
    //if (container->m_tetrahedronEdge.size()>0){
    container->getTetrahedronEdgeShellArray();
    //}

    /// force the creation of the Tetrahedron Triangle Shell array to detect isolated triangles
    //if (container->m_tetrahedronTriangle.size()>0){
    container->getTetrahedronTriangleShellArray();
    //}


    /// force the creation of the Tetrahedron Shell array to detect isolated vertices
    container->getTetrahedronVertexShellArray();

    container->getTriangleVertexShellArray();
    container->getTriangleEdgeShellArray();

    container->getEdgeVertexShellArray();


    //}

    if (container->m_tetrahedron.size()>0)
    {

        sofa::helper::vector<unsigned int> triangleToBeRemoved;
        sofa::helper::vector<unsigned int> edgeToBeRemoved;
        sofa::helper::vector<unsigned int> vertexToBeRemoved;

        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            Tetrahedron &t = container->m_tetrahedron[ indices[i] ];
            // first check that the tetrahedron vertex shell array has been initialized
            if (container->m_tetrahedronVertexShell.size()>0)
            {

                sofa::helper::vector< unsigned int > &shell0 = container->m_tetrahedronVertexShell[ t[0] ];
                shell0.erase(remove(shell0.begin(), shell0.end(), indices[i]), shell0.end());
                if ((removeIsolatedItems) && (shell0.size()==0))
                {
                    vertexToBeRemoved.push_back(t[0]);
                }

                sofa::helper::vector< unsigned int > &shell1 = container->m_tetrahedronVertexShell[ t[1] ];
                shell1.erase(remove(shell1.begin(), shell1.end(), indices[i]), shell1.end());
                if ((removeIsolatedItems) && (shell1.size()==0))
                {
                    vertexToBeRemoved.push_back(t[1]);
                }

                sofa::helper::vector< unsigned int > &shell2 = container->m_tetrahedronVertexShell[ t[2] ];
                shell2.erase(remove(shell2.begin(), shell2.end(), indices[i]), shell2.end());
                if ((removeIsolatedItems) && (shell2.size()==0))
                {
                    vertexToBeRemoved.push_back(t[2]);
                }

                sofa::helper::vector< unsigned int > &shell3 = container->m_tetrahedronVertexShell[ t[3] ];
                shell3.erase(remove(shell3.begin(), shell3.end(), indices[i]), shell3.end());
                if ((removeIsolatedItems) && (shell3.size()==0))
                {
                    vertexToBeRemoved.push_back(t[3]);
                }

            }


            /** first check that the tetrahedron edge shell array has been initialized */
            if (container->m_tetrahedronEdgeShell.size()>0)
            {

                sofa::helper::vector< unsigned int > &shell0 = container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][0]];
                shell0.erase(remove(shell0.begin(), shell0.end(), indices[i]), shell0.end());
                if ((removeIsolatedItems) && (shell0.size()==0))
                    edgeToBeRemoved.push_back(container->m_tetrahedronEdge[indices[i]][0]);

                sofa::helper::vector< unsigned int > &shell1 = container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][1]];
                shell1.erase(remove(shell1.begin(), shell1.end(), indices[i]), shell1.end());
                if ((removeIsolatedItems) && (shell1.size()==0))
                    edgeToBeRemoved.push_back(container->m_tetrahedronEdge[indices[i]][1]);

                sofa::helper::vector< unsigned int > &shell2 = container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][2]];
                shell2.erase(remove(shell2.begin(), shell2.end(), indices[i]), shell2.end());
                if ((removeIsolatedItems) && (shell2.size()==0))
                    edgeToBeRemoved.push_back(container->m_tetrahedronEdge[indices[i]][2]);

                sofa::helper::vector< unsigned int > &shell3 = container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][3]];
                shell3.erase(remove(shell3.begin(), shell3.end(), indices[i]), shell3.end());
                if ((removeIsolatedItems) && (shell3.size()==0))
                    edgeToBeRemoved.push_back(container->m_tetrahedronEdge[indices[i]][3]);

                sofa::helper::vector< unsigned int > &shell4 = container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][4]];
                shell4.erase(remove(shell4.begin(), shell4.end(), indices[i]), shell4.end());
                if ((removeIsolatedItems) && (shell4.size()==0))
                    edgeToBeRemoved.push_back(container->m_tetrahedronEdge[indices[i]][4]);

                sofa::helper::vector< unsigned int > &shell5 = container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][5]];
                shell5.erase(remove(shell5.begin(), shell5.end(), indices[i]), shell5.end());
                if ((removeIsolatedItems) && (shell5.size()==0))
                    edgeToBeRemoved.push_back(container->m_tetrahedronEdge[indices[i]][5]);

            }


            /** first check that the tetrahedron triangle shell array has been initialized */
            if (container->m_tetrahedronTriangleShell.size()>0)
            {
                sofa::helper::vector< unsigned int > &shell0 = container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][0]];
                shell0.erase(remove(shell0.begin(), shell0.end(), indices[i]), shell0.end());
                if ((removeIsolatedItems) && (shell0.size()==0))
                    triangleToBeRemoved.push_back(container->m_tetrahedronTriangle[indices[i]][0]);

                sofa::helper::vector< unsigned int > &shell1 = container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][1]];
                shell1.erase(remove(shell1.begin(), shell1.end(), indices[i]), shell1.end());
                if ((removeIsolatedItems) && (shell1.size()==0))
                    triangleToBeRemoved.push_back(container->m_tetrahedronTriangle[indices[i]][1]);

                sofa::helper::vector< unsigned int > &shell2 = container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][2]];
                shell2.erase(remove(shell2.begin(), shell2.end(), indices[i]), shell2.end());
                if ((removeIsolatedItems) && (shell2.size()==0))
                    triangleToBeRemoved.push_back(container->m_tetrahedronTriangle[indices[i]][2]);

                sofa::helper::vector< unsigned int > &shell3 = container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][3]];
                shell3.erase(remove(shell3.begin(), shell3.end(), indices[i]), shell3.end());
                if ((removeIsolatedItems) && (shell3.size()==0))
                    triangleToBeRemoved.push_back(container->m_tetrahedronTriangle[indices[i]][3]);

            }

            // removes the tetrahedron from the tetrahedronArray
            container->m_tetrahedron[ indices[i] ] = container->m_tetrahedron[ container->m_tetrahedron.size() - 1 ]; // overwriting with last valid value.


            if (container->m_tetrahedronEdge.size()>0)
            {
                // removes the tetrahedronEdges from the tetrahedronEdgeArray
                container->m_tetrahedronEdge[ indices[i] ] = container->m_tetrahedronEdge[ container->m_tetrahedron.size() - 1 ]; // overwriting with last valid value.
                container->m_tetrahedronEdge.resize( container->m_tetrahedronEdge.size() - 1 ); // resizing to erase multiple occurence of the edge.
            }

            if (container->m_tetrahedronTriangle.size()>0)
            {
                // removes the tetrahedronTriangles from the tetrahedronTriangleArray
                container->m_tetrahedronTriangle[ indices[i] ] = container->m_tetrahedronTriangle[ container->m_tetrahedron.size() - 1 ]; // overwriting with last valid value.
                container->m_tetrahedronTriangle.resize( container->m_tetrahedronTriangle.size() - 1 ); // resizing to erase multiple occurence of the triangle.
            }
            container->m_tetrahedron.resize( container->m_tetrahedron.size() - 1 ); // resizing to erase multiple occurence of the edge.


            // now updates the shell information of the edge formely at the end of the array
            // first check that the edge shell array has been initialized
            if ( indices[i] < container->m_tetrahedron.size() )
            {
                unsigned int oldTetrahedronIndex=container->m_tetrahedron.size();
                t = container->m_tetrahedron[ indices[i] ];
                if (container->m_tetrahedronVertexShell.size()>0)
                {
                    sofa::helper::vector< unsigned int > &shell0 = container->m_tetrahedronVertexShell[ t[0] ];
                    replace(shell0.begin(), shell0.end(), oldTetrahedronIndex, indices[i]);
                    sort(shell0.begin(), shell0.end());

                    sofa::helper::vector< unsigned int > &shell1 = container->m_tetrahedronVertexShell[ t[1] ];
                    replace(shell1.begin(), shell1.end(), oldTetrahedronIndex, indices[i]);
                    sort(shell1.begin(), shell1.end());

                    sofa::helper::vector< unsigned int > &shell2 = container->m_tetrahedronVertexShell[ t[2] ];
                    replace(shell2.begin(), shell2.end(), oldTetrahedronIndex, indices[i]);
                    sort(shell2.begin(), shell2.end());

                    sofa::helper::vector< unsigned int > &shell3 = container->m_tetrahedronVertexShell[ t[3] ];
                    replace(shell3.begin(), shell3.end(), oldTetrahedronIndex, indices[i]);
                    sort(shell3.begin(), shell3.end());
                }

                if (container->m_tetrahedronEdgeShell.size()>0)
                {
                    sofa::helper::vector< unsigned int > &shell0 =  container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][0]];
                    replace(shell0.begin(), shell0.end(), oldTetrahedronIndex, indices[i]);
                    sort(shell0.begin(), shell0.end());

                    sofa::helper::vector< unsigned int > &shell1 =  container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][1]];
                    replace(shell1.begin(), shell1.end(), oldTetrahedronIndex, indices[i]);
                    sort(shell1.begin(), shell1.end());

                    sofa::helper::vector< unsigned int > &shell2 =  container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][2]];
                    replace(shell2.begin(), shell2.end(), oldTetrahedronIndex, indices[i]);
                    sort(shell2.begin(), shell2.end());

                    sofa::helper::vector< unsigned int > &shell3 =  container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][3]];
                    replace(shell3.begin(), shell3.end(), oldTetrahedronIndex, indices[i]);
                    sort(shell3.begin(), shell3.end());

                    sofa::helper::vector< unsigned int > &shell4 =  container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][4]];
                    replace(shell4.begin(), shell4.end(), oldTetrahedronIndex, indices[i]);
                    sort(shell4.begin(), shell4.end());

                    sofa::helper::vector< unsigned int > &shell5 =  container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][5]];
                    replace(shell5.begin(), shell5.end(), oldTetrahedronIndex, indices[i]);
                    sort(shell5.begin(), shell5.end());
                }

                if (container->m_tetrahedronTriangleShell.size()>0)
                {
                    sofa::helper::vector< unsigned int > &shell0 =  container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][0]];
                    replace(shell0.begin(), shell0.end(), oldTetrahedronIndex, indices[i]);
                    sort(shell0.begin(), shell0.end());

                    sofa::helper::vector< unsigned int > &shell1 =  container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][1]];
                    replace(shell1.begin(), shell1.end(), oldTetrahedronIndex, indices[i]);
                    sort(shell1.begin(), shell1.end());

                    sofa::helper::vector< unsigned int > &shell2 =  container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][2]];
                    replace(shell2.begin(), shell2.end(), oldTetrahedronIndex, indices[i]);
                    sort(shell2.begin(), shell2.end());

                    sofa::helper::vector< unsigned int > &shell3 =  container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][3]];
                    replace(shell3.begin(), shell3.end(), oldTetrahedronIndex, indices[i]);
                    sort(shell3.begin(), shell3.end());

                }

            }
        }


        if ( (triangleToBeRemoved.size()>0) || (edgeToBeRemoved.size()>0) || (vertexToBeRemoved.size()>0))   //
        {

            if (triangleToBeRemoved.size()>0)
            {
                /// warn that triangles will be deleted
                this->removeTrianglesWarning(triangleToBeRemoved);
            }


            if (edgeToBeRemoved.size()>0)
            {
                /// warn that edges will be deleted
                this->removeEdgesWarning(edgeToBeRemoved);
            }

            //if (vertexToBeRemoved.size()>0){
            //	this->removePointsWarning(vertexToBeRemoved);
            //}

            /// propagate to all components

            topology->propagateTopologicalChanges();


            if (triangleToBeRemoved.size()>0)
            {
                /// actually remove triangles without looking for isolated vertices

                this->removeTrianglesProcess(triangleToBeRemoved,false,false);

            }

            if (edgeToBeRemoved.size()>0)
            {
                /// actually remove edges without looking for isolated vertices
                this->removeEdgesProcess(edgeToBeRemoved,false);
            }


            if (vertexToBeRemoved.size()>0)
            {
                this->removePointsWarning(vertexToBeRemoved);
                topology->propagateTopologicalChanges();
            }

            if (vertexToBeRemoved.size()>0)
            {
                this->removePointsProcess(vertexToBeRemoved);
            }


        }


    }
}



template<class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::addPointsProcess(const unsigned int nPoints,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs, const bool addDOF)
{
    // start by calling the standard method.
    TriangleSetTopologyModifier< DataTypes >::addPointsProcess( nPoints, ancestors, baryCoefs, addDOF );

    // now update the local container structures.
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_tetrahedronVertexShell.resize( container->m_tetrahedronVertexShell.size() + nPoints );
}

template<class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::addNewPoint(unsigned int i, const sofa::helper::vector< double >& x)
{
    // start by calling the standard method.
    TriangleSetTopologyModifier< DataTypes >::addNewPoint(i,x);

    // now update the local container structures.
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_tetrahedronVertexShell.resize( i+1 );
}


template<class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    // start by calling the standard method.
    TriangleSetTopologyModifier< DataTypes >::addEdgesProcess( edges );

    // now update the local container structures.
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_tetrahedronEdgeShell.resize( container->m_tetrahedronEdgeShell.size() + edges.size() );
}

template<class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::addTrianglesProcess(const sofa::helper::vector< Triangle > &triangles)
{
    // start by calling the standard method.
    TriangleSetTopologyModifier< DataTypes >::addTrianglesProcess( triangles );

    // now update the local container structures.
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_tetrahedronTriangleShell.resize( container->m_tetrahedronTriangleShell.size() + triangles.size() );
}



template< class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::removePointsProcess( sofa::helper::vector<unsigned int> &indices, const bool removeDOF)
{
    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)

    // now update the local container structures
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    //if (container->m_tetrahedron.size()>0)
    container->getTetrahedronVertexShellArray();

    // start by calling the standard method.
    TriangleSetTopologyModifier< DataTypes >::removePointsProcess(  indices, removeDOF );

    int vertexIndex;

    //assert(container->m_tetrahedronVertexShell.size()>0);

    unsigned int lastPoint = container->m_tetrahedronVertexShell.size() - 1;

    for (unsigned int i = 0; i < indices.size(); ++i)
    {
        // updating the edges connected to the point replacing the removed one:
        // for all edges connected to the last point
        sofa::helper::vector<unsigned int>::iterator itt=container->m_tetrahedronVertexShell[lastPoint].begin();
        for (; itt!=container->m_tetrahedronVertexShell[lastPoint].end(); ++itt)
        {

            vertexIndex=container->getVertexIndexInTetrahedron(container->m_tetrahedron[(*itt)],lastPoint);
            assert(vertexIndex!= -1);
            container->m_tetrahedron[(*itt)][(unsigned int)vertexIndex]=indices[i];
        }

        // updating the edge shell itself (change the old index for the new one)
        container->m_tetrahedronVertexShell[ indices[i] ] = container->m_tetrahedronVertexShell[ lastPoint ];

        --lastPoint;
    }

    container->m_tetrahedronVertexShell.resize( container->m_tetrahedronVertexShell.size() - indices.size() );
}

template< class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,const bool removeIsolatedItems)
{

    // now update the local container structures
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    //if (container->m_tetrahedronEdge.size()>0)
    container->getTetrahedronEdgeShellArray();

    // start by calling the standard method.
    TriangleSetTopologyModifier< DataTypes >::removeEdgesProcess(  indices, removeIsolatedItems );

    if (container->m_tetrahedronEdge.size()>0)
    {

        unsigned int edgeIndex;
        unsigned int lastEdge = container->m_tetrahedronEdgeShell.size() - 1;

        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            sofa::helper::vector<unsigned int>::iterator itt=container->m_tetrahedronEdgeShell[lastEdge].begin();
            for (; itt!=container->m_tetrahedronEdgeShell[lastEdge].end(); ++itt)
            {

                edgeIndex=container->getEdgeIndexInTetrahedron(container->m_tetrahedronEdge[(*itt)],lastEdge);
                assert((int)edgeIndex!= -1);
                container->m_tetrahedronEdge[(*itt)][(unsigned int)edgeIndex]=indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            container->m_tetrahedronEdgeShell[ indices[i] ] = container->m_tetrahedronEdgeShell[ lastEdge ];

            --lastEdge;
        }

        container->m_tetrahedronEdgeShell.resize( container->m_tetrahedronEdgeShell.size() - indices.size() );
    }

    /*
    if (container->m_tetrahedronEdgeShell.size()>0) {
      unsigned int lastEdge = container->m_tetrahedronEdgeShell.size() - 1;

      for (unsigned int i = 0; i < indices.size(); ++i)
        {
          // updating the edge shell itself (change the old index for the new one)
          container->m_tetrahedronEdgeShell[ indices[i] ] = container->m_tetrahedronEdgeShell[ lastEdge ];

          --lastEdge;
        }

      container->m_tetrahedronEdgeShell.resize( container->m_tetrahedronEdgeShell.size() - indices.size() );
    }
    */

}

template< class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::removeTrianglesProcess(  const sofa::helper::vector<unsigned int> &indices,const bool removeIsolatedEdges, const bool removeIsolatedPoints)
{
    // now update the local container structures
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    //if (container->m_tetrahedronTriangle.size()>0)
    container->getTetrahedronTriangleShellArray();

    // start by calling the standard method.

    TriangleSetTopologyModifier< DataTypes >::removeTrianglesProcess( indices, removeIsolatedEdges, removeIsolatedPoints );

    if (container->m_tetrahedronTriangle.size()>0)
    {
        unsigned int triangleIndex;
        unsigned int lastTriangle = container->m_tetrahedronTriangleShell.size() - 1;

        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            sofa::helper::vector<unsigned int>::iterator itt=container->m_tetrahedronTriangleShell[lastTriangle].begin();
            for (; itt!=container->m_tetrahedronTriangleShell[lastTriangle].end(); ++itt)
            {

                triangleIndex=container->getTriangleIndexInTetrahedron(container->m_tetrahedronTriangle[(*itt)],lastTriangle);
                assert((int)triangleIndex!= -1);
                container->m_tetrahedronTriangle[(*itt)][(unsigned int)triangleIndex]=indices[i];
            }

            // updating the triangle shell itself (change the old index for the new one)
            container->m_tetrahedronTriangleShell[ indices[i] ] = container->m_tetrahedronTriangleShell[ lastTriangle ];

            --lastTriangle;
        }

        container->m_tetrahedronTriangleShell.resize( container->m_tetrahedronTriangleShell.size() - indices.size() );
    }

}


template< class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index, const sofa::helper::vector<unsigned int> &inv_index, const bool renumberDOF)
{
    // start by calling the standard method
    TriangleSetTopologyModifier< DataTypes >::renumberPointsProcess( index, inv_index, renumberDOF );

    // now update the local container structures.
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    sofa::helper::vector< sofa::helper::vector< unsigned int > > tetrahedronVertexShell_cp = container->m_tetrahedronVertexShell;
    for (unsigned int i = 0; i < index.size(); ++i)
    {
        container->m_tetrahedronVertexShell[i] = tetrahedronVertexShell_cp[ index[i] ];
    }

    for (unsigned int i = 0; i < container->m_tetrahedron.size(); ++i)
    {
        container->m_tetrahedron[i][0]  = inv_index[ container->m_tetrahedron[i][0]  ];
        container->m_tetrahedron[i][1]  = inv_index[ container->m_tetrahedron[i][1]  ];
        container->m_tetrahedron[i][2]  = inv_index[ container->m_tetrahedron[i][2]  ];
        container->m_tetrahedron[i][3]  = inv_index[ container->m_tetrahedron[i][3]  ];
    }


}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////TetrahedronSetGeometryAlgorithms//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////


template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::removeTetrahedra(sofa::helper::vector< unsigned int >& tetrahedra)
{
    TetrahedronSetTopology< DataTypes > *topology = dynamic_cast<TetrahedronSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyModifier< DataTypes >* modifier  = static_cast< TetrahedronSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(modifier != 0);
    /// add the topological changes in the queue

    //TetrahedronSetTopologyContainer * container = static_cast< TetrahedronSetTopologyContainer* >(topology->getTopologyContainer());

    modifier->removeTetrahedraWarning(tetrahedra);


    // inform other objects that the triangles are going to be removed

    topology->propagateTopologicalChanges();

    // now destroy the old tetrahedra.

    modifier->removeTetrahedraProcess(  tetrahedra ,true);

    assert(topology->getTetrahedronSetTopologyContainer()->checkTopology());

}

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::removeItems(sofa::helper::vector< unsigned int >& items)
{
    removeTetrahedra(items);
}

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::RemoveTetraBall(unsigned int ind_ta, unsigned int ind_tb)
{

    // Access the topology
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);

    sofa::helper::vector<unsigned int> init_indices;
    sofa::helper::vector<unsigned int> &indices = init_indices;
    topology->getTetrahedronSetGeometryAlgorithms()->getTetraInBall(ind_ta, ind_tb, indices);
    removeTetrahedra(indices);

    //cout<<"INFO, number to remove = "<< indices.size() <<endl;
}

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::writeMSH(const char *filename)
{

    TetrahedronSetTopology< DataTypes > *topology = dynamic_cast<TetrahedronSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyModifier< DataTypes >* modifier  = static_cast< TetrahedronSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(modifier != 0);

    modifier->writeMSHfile(filename);
}

template<class DataTypes>
void  TetrahedronSetTopologyAlgorithms<DataTypes>::renumberPoints( const sofa::helper::vector<unsigned int> &index, const sofa::helper::vector<unsigned int> &inv_index)
{

    TetrahedronSetTopology< DataTypes > *topology = dynamic_cast<TetrahedronSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyModifier< DataTypes >* modifier  = static_cast< TetrahedronSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(modifier != 0);
    /// add the topological changes in the queue
    modifier->renumberPointsWarning(index, inv_index);
    // inform other objects that the triangles are going to be removed
    topology->propagateTopologicalChanges();
    // now renumber the points
    modifier->renumberPointsProcess(index, inv_index);

    assert(topology->getTetrahedronSetTopologyContainer()->checkTopology());
}

/// Cross product for 3-elements vectors.
template<typename real>
inline real tripleProduct(const Vec<3,real>& a, const Vec<3,real>& b,const Vec<3,real> &c)
{
    return dot(a,cross(b,c));
}

/// area from 2-elements vectors.
template <typename real>
inline real tripleProduct(const Vec<2,real>& , const Vec<2,real>& ,const Vec<2,real> &)
{
    assert(false);
    return (real)0;
}
/// area for 1-elements vectors.
template <typename real>
inline real tripleProduct(const Vec<1,real>& , const Vec<1,real>& ,const Vec<1,real> &)
{
    assert(false);
    return (real)0;
}

template< class DataTypes>
typename DataTypes::Real TetrahedronSetGeometryAlgorithms< DataTypes >::computeTetrahedronVolume( const unsigned int i) const
{
    TetrahedronSetTopology< DataTypes > *topology = dynamic_cast<TetrahedronSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast< TetrahedronSetTopologyContainer* >(topology->getTopologyContainer());
    const Tetrahedron &t=container->getTetrahedron(i);
    const VecCoord& p = *topology->getDOF()->getX();
    Real volume=(Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    return volume;
}
template< class DataTypes>
typename DataTypes::Real TetrahedronSetGeometryAlgorithms< DataTypes >::computeRestTetrahedronVolume( const unsigned int i) const
{
    TetrahedronSetTopology< DataTypes > *topology = dynamic_cast<TetrahedronSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast< TetrahedronSetTopologyContainer* >(topology->getTopologyContainer());
    const Tetrahedron &t=container->getTetrahedron(i);
    const VecCoord& p = *topology->getDOF()->getX0();
    Real volume=(Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    return volume;

}

/// computes the edge length of all edges are store in the array interface
template<class DataTypes>
void TetrahedronSetGeometryAlgorithms<DataTypes>::computeTetrahedronVolume( BasicArrayInterface<Real> &ai) const
{
    TetrahedronSetTopology< DataTypes > *topology = dynamic_cast<TetrahedronSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast< TetrahedronSetTopologyContainer* >(topology->getTopologyContainer());
    const sofa::helper::vector<Tetrahedron> &ta=container->getTetrahedronArray();
    const typename DataTypes::VecCoord& p = *topology->getDOF()->getX();
    unsigned int i;
    for (i=0; i<ta.size(); ++i)
    {
        const Tetrahedron &t=ta[i];
        ai[i]=(Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    }
}

/// Finds the indices of all tetrahedra in the ball of center ind_ta and of radius dist(ind_ta, ind_tb)
template<class DataTypes>
void TetrahedronSetGeometryAlgorithms< DataTypes >::getTetraInBall(unsigned int ind_ta, unsigned int ind_tb,  sofa::helper::vector<unsigned int> &indices)
{

    TetrahedronSetTopology< DataTypes > *topology = static_cast<TetrahedronSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast< TetrahedronSetTopologyContainer* >(topology->getTopologyContainer());
    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();

    const Tetrahedron &ta=container->getTetrahedron(ind_ta);
    const Tetrahedron &tb=container->getTetrahedron(ind_tb);

    const typename DataTypes::Coord& ca=(vect_c[ta[0]]+vect_c[ta[1]]+vect_c[ta[2]]+vect_c[ta[3]])*0.25;
    const typename DataTypes::Coord& cb=(vect_c[tb[0]]+vect_c[tb[1]]+vect_c[tb[2]]+vect_c[tb[3]])*0.25;
    Vec<3,Real> pa;
    Vec<3,Real> pb;
    pa[0] = (Real) (ca[0]); pa[1] = (Real) (ca[1]); pa[2] = (Real) (ca[2]);
    pb[0] = (Real) (cb[0]); pb[1] = (Real) (cb[1]); pb[2] = (Real) (cb[2]);

    Real d = (pa-pb)*(pa-pb);

    unsigned int t_test=ind_ta;
    indices.push_back(t_test);

    std::map<unsigned int, unsigned int> IndexMap;
    IndexMap.clear();
    IndexMap[t_test]=0;

    sofa::helper::vector<unsigned int> ind2test;
    ind2test.push_back(t_test);
    sofa::helper::vector<unsigned int> ind2ask;
    ind2ask.push_back(t_test);

    while(ind2test.size()>0)
    {

        ind2test.clear();
        for (unsigned int t=0; t<ind2ask.size(); t++)
        {

            unsigned int ind_t = ind2ask[t];
            sofa::component::topology::TetrahedronTriangles adjacent_triangles = container->getTetrahedronTriangles(ind_t);

            for (unsigned int i=0; i<adjacent_triangles.size(); i++)
            {
                sofa::helper::vector< unsigned int > tetras_to_remove = container->getTetrahedronTriangleShell(adjacent_triangles[i]);

                if(tetras_to_remove.size()==2)
                {
                    if(tetras_to_remove[0]==ind_t)
                    {
                        t_test=tetras_to_remove[1];
                    }
                    else
                    {
                        t_test=tetras_to_remove[0];
                    }

                    std::map<unsigned int, unsigned int>::iterator iter_1 = IndexMap.find(t_test);
                    if(iter_1 == IndexMap.end())
                    {

                        IndexMap[t_test]=0;

                        const Tetrahedron &tc=container->getTetrahedron(t_test);
                        const typename DataTypes::Coord& cc=(vect_c[tc[0]]+vect_c[tc[1]]+vect_c[tc[2]]+vect_c[tc[3]])*0.25;
                        Vec<3,Real> pc;
                        pc[0] = (Real) (cc[0]); pc[1] = (Real) (cc[1]); pc[2] = (Real) (cc[2]);

                        Real d_test = (pa-pc)*(pa-pc);

                        if(d_test<d)
                        {
                            ind2test.push_back(t_test);
                            indices.push_back(t_test);
                        }
                    }
                }
            }
        }

        ind2ask.clear();
        for (unsigned int t=0; t<ind2test.size(); t++)
        {
            ind2ask.push_back(ind2test[t]);
        }

    }

    return;

}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TETEAHEDRONSETTOPOLOGY_INL
