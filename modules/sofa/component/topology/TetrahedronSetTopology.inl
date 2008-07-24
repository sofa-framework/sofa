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

const unsigned int tetrahedronEdgeArray[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};

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
///////////////////////////TetrahedronSetTopologyAlgorithms//////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::removeTetrahedra(sofa::helper::vector< unsigned int >& tetrahedra)
{
    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyModifier< DataTypes >* modifier  = topology->getTetrahedronSetTopologyModifier();

    modifier->removeTetrahedraWarning(tetrahedra);

    // inform other objects that the triangles are going to be removed
    topology->propagateTopologicalChanges();

    // now destroy the old tetrahedra.
    modifier->removeTetrahedraProcess(  tetrahedra ,true);

    topology->getTetrahedronSetTopologyContainer()->checkTopology();
}

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::removeItems(sofa::helper::vector< unsigned int >& items)
{
    removeTetrahedra(items);
}

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::RemoveTetraBall(unsigned int ind_ta, unsigned int ind_tb)
{
    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();

    sofa::helper::vector<unsigned int> init_indices;
    sofa::helper::vector<unsigned int> &indices = init_indices;
    topology->getTetrahedronSetGeometryAlgorithms()->getTetraInBall(ind_ta, ind_tb, indices);
    removeTetrahedra(indices);

    //cout<<"INFO, number to remove = "<< indices.size() <<endl;
}

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::writeMSH(const char *filename)
{
    getTetrahedronSetTopology()->getTetrahedronSetTopologyModifier()->writeMSHfile(filename);
}

template<class DataTypes>
void  TetrahedronSetTopologyAlgorithms<DataTypes>::renumberPoints( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index)
{
    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyModifier< DataTypes >* modifier  = topology->getTetrahedronSetTopologyModifier();

    /// add the topological changes in the queue
    modifier->renumberPointsWarning(index, inv_index);
    // inform other objects that the triangles are going to be removed
    topology->propagateTopologicalChanges();
    // now renumber the points
    modifier->renumberPointsProcess(index, inv_index);

    topology->getTetrahedronSetTopologyContainer()->checkTopology();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////TetrahedronSetGeometryAlgorithms//////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template< class DataTypes>
typename DataTypes::Real TetrahedronSetGeometryAlgorithms< DataTypes >::computeTetrahedronVolume( const unsigned int i) const
{
    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyContainer * container = topology->getTetrahedronSetTopologyContainer();

    const Tetrahedron &t = container->getTetrahedron(i);
    const VecCoord& p = *topology->getDOF()->getX();
    Real volume = (Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    return volume;
}

template< class DataTypes>
typename DataTypes::Real TetrahedronSetGeometryAlgorithms< DataTypes >::computeRestTetrahedronVolume( const unsigned int i) const
{
    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyContainer * container = topology->getTetrahedronSetTopologyContainer();

    const Tetrahedron &t=container->getTetrahedron(i);
    const VecCoord& p = *topology->getDOF()->getX0();
    Real volume = (Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    return volume;
}

/// computes the edge length of all edges are store in the array interface
template<class DataTypes>
void TetrahedronSetGeometryAlgorithms<DataTypes>::computeTetrahedronVolume( BasicArrayInterface<Real> &ai) const
{
    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyContainer * container = topology->getTetrahedronSetTopologyContainer();

    const sofa::helper::vector<Tetrahedron> &ta = container->getTetrahedronArray();
    const typename DataTypes::VecCoord& p = *topology->getDOF()->getX();
    for (unsigned int i=0; i<ta.size(); ++i)
    {
        const Tetrahedron &t = ta[i];
        ai[i] = (Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    }
}

/// Finds the indices of all tetrahedra in the ball of center ind_ta and of radius dist(ind_ta, ind_tb)
template<class DataTypes>
void TetrahedronSetGeometryAlgorithms< DataTypes >::getTetraInBall(unsigned int ind_ta, unsigned int ind_tb,
        sofa::helper::vector<unsigned int> &indices)
{
    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyContainer * container = topology->getTetrahedronSetTopologyContainer();

    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();

    const Tetrahedron &ta=container->getTetrahedron(ind_ta);
    const Tetrahedron &tb=container->getTetrahedron(ind_tb);

    const typename DataTypes::Coord& ca=(vect_c[ta[0]]+vect_c[ta[1]]+vect_c[ta[2]]+vect_c[ta[3]])*0.25;
    const typename DataTypes::Coord& cb=(vect_c[tb[0]]+vect_c[tb[1]]+vect_c[tb[2]]+vect_c[tb[3]])*0.25;
    Vec<3,Real> pa;
    Vec<3,Real> pb;
    pa[0] = (Real) (ca[0]);
    pa[1] = (Real) (ca[1]);
    pa[2] = (Real) (ca[2]);
    pb[0] = (Real) (cb[0]);
    pb[1] = (Real) (cb[1]);
    pb[2] = (Real) (cb[2]);

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
                        const typename DataTypes::Coord& cc = (vect_c[tc[0]]
                                + vect_c[tc[1]]
                                + vect_c[tc[2]]
                                + vect_c[tc[3]]) * 0.25;
                        Vec<3,Real> pc;
                        pc[0] = (Real) (cc[0]);
                        pc[1] = (Real) (cc[1]);
                        pc[2] = (Real) (cc[2]);

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

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////TetrahedronSetTopologyModifier//////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class DataTypes>
class TetrahedronSetTopologyLoader : public PointSetTopologyLoader<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;

    VecCoord pointArray;
    TetrahedronSetTopologyModifier<DataTypes> *tstm;

    TetrahedronSetTopologyLoader(TetrahedronSetTopologyModifier<DataTypes> *tm)
        : PointSetTopologyLoader<DataTypes>(),
          tstm(tm)
    { }

    virtual void addTetra(int p1, int p2, int p3,int p4)
    {
        tstm->addTetrahedron(Tetrahedron(helper::make_array<unsigned int>((unsigned int)p1,
                (unsigned int)p2,
                (unsigned int) p3,
                (unsigned int) p4)));
    }
};

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

    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyContainer * container = topology->getTetrahedronSetTopologyContainer();

    PointSetTopology< Vec3Types >* psp = dynamic_cast< PointSetTopology< Vec3Types >* >( topology );

    Vec3Types::VecCoord &p = *psp->getDOF()->getX();

    myfile << "$NOD\n";
    myfile << container->getNumberOfVertices() <<"\n";

    for (unsigned int i=0; i<p.size(); ++i)
    {
        double x = (double) p[i][0];
        double y = (double) p[i][1];
        double z = (double) p[i][2];

        myfile << i+1 << " " << x << " " << y << " " << z <<"\n";
    }

    myfile << "$ENDNOD\n";
    myfile << "$ELM\n";

    const sofa::helper::vector<Tetrahedron> &tea = container->getTetrahedronArray();

    myfile << tea.size() <<"\n";

    for (unsigned int i=0; i<tea.size(); ++i)
    {
        myfile << i+1 << " 4 1 1 4 " << tea[i][0]+1 << " " << tea[i][1]+1 << " " << tea[i][2]+1 << " " << tea[i][3]+1 <<"\n";
    }

    myfile << "$ENDELM\n";

    myfile.close();
}

template<class DataTypes>
void TetrahedronSetTopologyModifier<DataTypes>::addTetrahedron(Tetrahedron t)
{
    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyContainer * container = topology->getTetrahedronSetTopologyContainer();

    // check if the 3 vertices are different
    assert(t[0]!=t[1]);
    assert(t[0]!=t[2]);
    assert(t[0]!=t[3]);
    assert(t[1]!=t[2]);
    assert(t[1]!=t[3]);
    assert(t[2]!=t[3]);

    // check if there already exists a tetrahedron with the same indices
    // assert(container->getTetrahedronIndex(t[0], t[1], t[2], t[3])== -1);

    unsigned int tetrahedronIndex = container->m_tetrahedron.size();

    if (container->hasTetrahedronTriangles())
    {
        for (unsigned int j=0; j<4; ++j)
        {
            int triangleIndex = container->getTriangleIndex(t[(j+1)%4], t[(j+2)%4], t[(j+3)%4]);

            if(triangleIndex == -1)
            {
                // first create the traingle
                sofa::helper::vector< Triangle > v;
                Triangle e1 (t[(j+1)%4],t[(j+2)%4],t[(j+3)%4]);
                v.push_back(e1);

                addTrianglesProcess((const sofa::helper::vector< Triangle > &) v);

                triangleIndex = container->getTriangleIndex(t[(j+1)%4], t[(j+2)%4], t[(j+3)%4]);

                sofa::helper::vector< unsigned int > triangleIndexList;
                triangleIndexList.push_back(triangleIndex);
                this->addTrianglesWarning( v.size(), v, triangleIndexList);
            }

            container->m_tetrahedronTriangle.resize(triangleIndex+1);
            container->m_tetrahedronTriangle[tetrahedronIndex][j]= triangleIndex;
        }
    }

    if (container->hasTetrahedronEdges())
    {
        for (unsigned int j=0; j<6; ++j)
        {
            int edgeIndex=container->getEdgeIndex(tetrahedronEdgeArray[j][0],
                    tetrahedronEdgeArray[j][1]);
            assert(edgeIndex!= -1);

            container->m_tetrahedronEdge.resize(edgeIndex+1);
            container->m_tetrahedronEdge[tetrahedronIndex][j]= edgeIndex;
        }
    }

    if (container->hasTetrahedronVertexShell())
    {
        for (unsigned int j=0; j<4; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = container->getTetrahedronVertexShellForModification( t[j] );
            shell.push_back( tetrahedronIndex );
        }
    }

    if (container->hasTetrahedronEdgeShell())
    {
        for (unsigned int j=0; j<6; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = container->m_tetrahedronEdgeShell[container->m_tetrahedronEdge[tetrahedronIndex][j]];
            shell.push_back( tetrahedronIndex );
        }
    }

    if (container->hasTetrahedronTriangleShell())
    {
        for (unsigned int j=0; j<4; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = container->m_tetrahedronTriangleShell[container->m_tetrahedronTriangle[tetrahedronIndex][j]];
            shell.push_back( tetrahedronIndex );
        }
    }

    container->m_tetrahedron.push_back(t);
}

template<class DataTypes>
void TetrahedronSetTopologyModifier<DataTypes>::addTetrahedraProcess(const sofa::helper::vector< Tetrahedron > &tetrahedra)
{
    for (unsigned int i = 0; i < tetrahedra.size(); ++i)
    {
        addTetrahedron(tetrahedra[i]);
    }
}

template<class DataTypes>
void TetrahedronSetTopologyModifier<DataTypes>::addTetrahedraWarning(const unsigned int nTetrahedra,
        const sofa::helper::vector< Tetrahedron >& tetrahedraList,
        const sofa::helper::vector< unsigned int >& tetrahedraIndexList)
{
    // Warning that tetrahedra just got created
    TetrahedraAdded *e = new TetrahedraAdded(nTetrahedra, tetrahedraList, tetrahedraIndexList);
    this->addTopologyChange(e);
}

template<class DataTypes>
void TetrahedronSetTopologyModifier<DataTypes>::addTetrahedraWarning(const unsigned int nTetrahedra,
        const sofa::helper::vector< Tetrahedron >& tetrahedraList,
        const sofa::helper::vector< unsigned int >& tetrahedraIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    // Warning that tetrahedra just got created
    TetrahedraAdded *e = new TetrahedraAdded(nTetrahedra, tetrahedraList, tetrahedraIndexList, ancestors, baryCoefs);
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
void TetrahedronSetTopologyModifier<DataTypes>::removeTetrahedraProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{
    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyContainer * container = topology->getTetrahedronSetTopologyContainer();

    bool removeIsolatedVertices = removeIsolatedItems;
    bool removeIsolatedEdges = removeIsolatedItems && container->hasEdges();
    bool removeIsolatedTriangles = removeIsolatedItems && container->hasTriangles();

    if(removeIsolatedVertices)
    {
        if(!container->hasTetrahedronVertexShell())
            container->createTetrahedronVertexShellArray();
    }

    if(removeIsolatedEdges)
    {
        if(!container->hasTetrahedronEdgeShell())
            container->createTetrahedronEdgeShellArray();
    }

    if(removeIsolatedTriangles)
    {
        if(!container->hasTetrahedronTriangleShell())
            container->createTetrahedronTriangleShellArray();
    }

    sofa::helper::vector<unsigned int> triangleToBeRemoved;
    sofa::helper::vector<unsigned int> edgeToBeRemoved;
    sofa::helper::vector<unsigned int> vertexToBeRemoved;

    unsigned int lastTetrahedron = container->getNumberOfTetrahedra() - 1;
    for (unsigned int i=0; i<indices.size(); ++i, --lastTetrahedron)
    {
        Tetrahedron &t = container->m_tetrahedron[ indices[i] ];
        Tetrahedron &h = container->m_tetrahedron[ lastTetrahedron ];

        // first check that the tetrahedron vertex shell array has been initialized
        if (container->hasTetrahedronVertexShell())
        {
            for(unsigned int j=0; j<4; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = container->m_tetrahedronVertexShell[ t[j] ];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if ((removeIsolatedItems) && (shell.empty()))
                    vertexToBeRemoved.push_back(t[j]);
            }
        }

        /** first check that the tetrahedron edge shell array has been initialized */
        if (container->hasTetrahedronEdgeShell())
        {
            for(unsigned int j=0; j<6; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][j]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if ((removeIsolatedItems) && (shell.empty()))
                    edgeToBeRemoved.push_back(container->m_tetrahedronEdge[indices[i]][j]);
            }
        }

        /** first check that the tetrahedron triangle shell array has been initialized */
        if (container->hasTetrahedronTriangleShell())
        {
            for(unsigned int j=0; j<4; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][j]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if ((removeIsolatedItems) && (shell.empty()))
                    triangleToBeRemoved.push_back(container->m_tetrahedronTriangle[indices[i]][j]);
            }
        }

        // now updates the shell information of the edge formely at the end of the array
        // first check that the edge shell array has been initialized
        if ( indices[i] < lastTetrahedron )
        {
            if (container->hasTetrahedronVertexShell())
            {
                for(unsigned int j=0; j<4; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell = container->m_tetrahedronVertexShell[ h[j] ];
                    replace(shell.begin(), shell.end(), lastTetrahedron, indices[i]);
                }
            }

            if (container->hasTetrahedronEdgeShell())
            {
                for(unsigned int j=0; j<6; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell =  container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[lastTetrahedron][j]];
                    replace(shell.begin(), shell.end(), lastTetrahedron, indices[i]);
                }
            }

            if (container->hasTetrahedronTriangleShell())
            {
                for(unsigned int j=0; j<4; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell =  container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[lastTetrahedron][j]];
                    replace(shell.begin(), shell.end(), lastTetrahedron, indices[i]);
                }
            }
        }

        if (container->hasTetrahedronTriangles())
        {
            // removes the tetrahedronTriangles from the tetrahedronTriangleArray
            container->m_tetrahedronTriangle[ indices[i] ] = container->m_tetrahedronTriangle[ lastTetrahedron ]; // overwriting with last valid value.
            container->m_tetrahedronTriangle.resize( lastTetrahedron ); // resizing to erase multiple occurence of the tetrahedron.
        }

        if (container->hasTetrahedronEdges())
        {
            // removes the tetrahedronEdges from the tetrahedronEdgeArray
            container->m_tetrahedronEdge[ indices[i] ] = container->m_tetrahedronEdge[ lastTetrahedron ]; // overwriting with last valid value.
            container->m_tetrahedronEdge.resize( lastTetrahedron ); // resizing to erase multiple occurence of the tetrahedron.
        }

        // removes the tetrahedron from the tetrahedronArray
        container->m_tetrahedron[ indices[i] ] = container->m_tetrahedron[ lastTetrahedron ]; // overwriting with last valid value.
        container->m_tetrahedron.resize( lastTetrahedron ); // resizing to erase multiple occurence of the tetrahedron.
    }

    if ( (!triangleToBeRemoved.empty()) || (!edgeToBeRemoved.empty()))
    {
        if (!triangleToBeRemoved.empty())
        {
            /// warn that triangles will be deleted
            this->removeTrianglesWarning(triangleToBeRemoved);
        }

        if (!edgeToBeRemoved.empty())
        {
            /// warn that edges will be deleted
            this->removeEdgesWarning(edgeToBeRemoved);
        }

        /// propagate to all components
        topology->propagateTopologicalChanges();

        if (!triangleToBeRemoved.empty())
        {
            /// actually remove triangles without looking for isolated vertices
            this->removeTrianglesProcess(triangleToBeRemoved, false, false);

        }

        if (!edgeToBeRemoved.empty())
        {
            /// actually remove edges without looking for isolated vertices
            this->removeEdgesProcess(edgeToBeRemoved, false);
        }
    }

    if (!vertexToBeRemoved.empty())
    {
        this->removePointsWarning(vertexToBeRemoved);
        topology->propagateTopologicalChanges();
        this->removePointsProcess(vertexToBeRemoved);
    }
}

template<class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::addPointsProcess(const unsigned int nPoints,
        const bool addDOF)
{
    // start by calling the parent's method.
    // TODO : only if triangles exist, otherwise call the EdgeSet or PointSet method respectively
    TriangleSetTopologyModifier< DataTypes >::addPointsProcess( nPoints, addDOF );

    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyContainer * container = topology->getTetrahedronSetTopologyContainer();

    container->m_tetrahedronVertexShell.resize( container->m_tetrahedronVertexShell.size() + nPoints );
}

template<class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::addPointsProcess(const unsigned int nPoints,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
        const bool addDOF)
{
    // start by calling the parent's method.
    // TODO : only if triangles exist, otherwise call the EdgeSet or PointSet method respectively
    TriangleSetTopologyModifier< DataTypes >::addPointsProcess( nPoints, ancestors, baryCoefs, addDOF );

    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyContainer * container = topology->getTetrahedronSetTopologyContainer();

    container->m_tetrahedronVertexShell.resize( container->m_tetrahedronVertexShell.size() + nPoints );
}

template<class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::addNewPoint(unsigned int i, const sofa::helper::vector< double >& x)
{
    // start by calling the parent's method.
    // TODO : only if triangles exist, otherwise call the EdgeSet or PointSet method respectively
    TriangleSetTopologyModifier< DataTypes >::addNewPoint(i,x);

    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyContainer * container = topology->getTetrahedronSetTopologyContainer();

    container->m_tetrahedronVertexShell.resize( i+1 );
}

template<class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    // start by calling the parent's method.
    // TODO : only if triangles exist, otherwise call the EdgeSet or PointSet method respectively
    TriangleSetTopologyModifier< DataTypes >::addEdgesProcess( edges );

    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyContainer * container = topology->getTetrahedronSetTopologyContainer();

    container->m_tetrahedronEdgeShell.resize( container->m_tetrahedronEdgeShell.size() + edges.size() );
}

template<class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::addTrianglesProcess(const sofa::helper::vector< Triangle > &triangles)
{
    // start by calling the parent's method.
    // TODO : only if triangles exist
    TriangleSetTopologyModifier< DataTypes >::addTrianglesProcess( triangles );

    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyContainer * container = topology->getTetrahedronSetTopologyContainer();

    container->m_tetrahedronTriangleShell.resize( container->m_tetrahedronTriangleShell.size() + triangles.size() );
}

template< class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::removePointsProcess( sofa::helper::vector<unsigned int> &indices,
        const bool removeDOF)
{
    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyContainer * container = topology->getTetrahedronSetTopologyContainer();

    if(!container->hasTetrahedronVertexShell())
    {
        container->createTetrahedronVertexShellArray();
    }

    unsigned int lastPoint = container->m_tetrahedronVertexShell.size() - 1;
    for (unsigned int i=0; i<indices.size(); ++i, --lastPoint)
    {
        // updating the edges connected to the point replacing the removed one:
        // for all edges connected to the last point
        for (sofa::helper::vector<unsigned int>::iterator itt=container->m_tetrahedronVertexShell[lastPoint].begin();
                itt!=container->m_tetrahedronVertexShell[lastPoint].end(); ++itt)
        {
            int vertexIndex = container->getVertexIndexInTetrahedron(container->m_tetrahedron[(*itt)],lastPoint);
            assert(vertexIndex!= -1);
            container->m_tetrahedron[(*itt)][(unsigned int)vertexIndex]=indices[i];
        }

        // updating the edge shell itself (change the old index for the new one)
        container->m_tetrahedronVertexShell[ indices[i] ] = container->m_tetrahedronVertexShell[ lastPoint ];
    }

    container->m_tetrahedronVertexShell.resize( container->m_tetrahedronVertexShell.size() - indices.size() );

    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    // call the parent's method.
    // TODO : only if triangles exist, otherwise call the EdgeSet or PointSet method respectively
    TriangleSetTopologyModifier< DataTypes >::removePointsProcess(  indices, removeDOF );
}

template< class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{
    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyContainer * container = topology->getTetrahedronSetTopologyContainer();

    if(!container->hasEdges()) // TODO : this method should only be called when edges exist
        return;

    if (container->hasTetrahedronEdges())
    {
        if(!container->hasTetrahedronEdgeShell())
            container->createTetrahedronEdgeShellArray();

        unsigned int lastEdge = container->getNumberOfEdges() - 1;
        for (unsigned int i=0; i<indices.size(); ++i, --lastEdge)
        {
            for (sofa::helper::vector<unsigned int>::iterator itt=container->m_tetrahedronEdgeShell[lastEdge].begin();
                    itt!=container->m_tetrahedronEdgeShell[lastEdge].end(); ++itt)
            {
                int edgeIndex=container->getEdgeIndexInTetrahedron(container->m_tetrahedronEdge[(*itt)],lastEdge);
                assert((int)edgeIndex!= -1);
                container->m_tetrahedronEdge[(*itt)][(unsigned int) edgeIndex]=indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            container->m_tetrahedronEdgeShell[ indices[i] ] = container->m_tetrahedronEdgeShell[ lastEdge ];
        }

        container->m_tetrahedronEdgeShell.resize( container->m_tetrahedronEdgeShell.size() - indices.size() );
    }

    // call the parent's method.
    // TODO : only if triangles exist, otherwise call the EdgeSet or PointSet method respectively
    TriangleSetTopologyModifier< DataTypes >::removeEdgesProcess( indices, removeIsolatedItems );
}

template< class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::removeTrianglesProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyContainer * container = topology->getTetrahedronSetTopologyContainer();

    if(!container->hasTriangles()) // TODO : this method should only be called when triangles exist
        return;

    if (container->hasTetrahedronTriangles())
    {
        if(!container->hasTetrahedronTriangleShell())
            container->createTetrahedronTriangleShellArray();

        unsigned int lastTriangle = container->m_tetrahedronTriangleShell.size() - 1;
        for (unsigned int i = 0; i < indices.size(); ++i, --lastTriangle)
        {
            for (sofa::helper::vector<unsigned int>::iterator itt=container->m_tetrahedronTriangleShell[lastTriangle].begin();
                    itt!=container->m_tetrahedronTriangleShell[lastTriangle].end(); ++itt)
            {
                int triangleIndex=container->getTriangleIndexInTetrahedron(container->m_tetrahedronTriangle[(*itt)],lastTriangle);
                assert((int)triangleIndex!= -1);
                container->m_tetrahedronTriangle[(*itt)][(unsigned int)triangleIndex] = indices[i];
            }

            // updating the triangle shell itself (change the old index for the new one)
            container->m_tetrahedronTriangleShell[ indices[i] ] = container->m_tetrahedronTriangleShell[ lastTriangle ];
        }

        container->m_tetrahedronTriangleShell.resize( container->m_tetrahedronTriangleShell.size() - indices.size() );
    }

    // call the parent's method.
    // TODO : only if triangles exist, otherwise call the EdgeSet or PointSet method respectively
    TriangleSetTopologyModifier< DataTypes >::removeTrianglesProcess( indices, removeIsolatedEdges, removeIsolatedPoints );
}

template< class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index,
        const bool renumberDOF)
{
    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyContainer * container = topology->getTetrahedronSetTopologyContainer();

    if(container->hasTetrahedronVertexShell())
    {
        sofa::helper::vector< sofa::helper::vector< unsigned int > > tetrahedronVertexShell_cp = container->m_tetrahedronVertexShell;
        for (unsigned int i = 0; i < index.size(); ++i)
        {
            container->m_tetrahedronVertexShell[i] = tetrahedronVertexShell_cp[ index[i] ];
        }
    }

    for (unsigned int i=0; i<container->m_tetrahedron.size(); ++i)
    {
        container->m_tetrahedron[i][0]  = inv_index[ container->m_tetrahedron[i][0]  ];
        container->m_tetrahedron[i][1]  = inv_index[ container->m_tetrahedron[i][1]  ];
        container->m_tetrahedron[i][2]  = inv_index[ container->m_tetrahedron[i][2]  ];
        container->m_tetrahedron[i][3]  = inv_index[ container->m_tetrahedron[i][3]  ];
    }

    // call the parent's method.
    // TODO : only if triangles exist, otherwise call the EdgeSet or PointSet method respectively
    TriangleSetTopologyModifier< DataTypes >::renumberPointsProcess( index, inv_index, renumberDOF );
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TETEAHEDRONSETTOPOLOGY_INL
