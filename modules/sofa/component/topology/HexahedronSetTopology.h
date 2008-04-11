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
#ifndef SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGY_H

#include <sofa/component/topology/QuadSetTopology.h>
#include <vector>
#include <map>

namespace sofa
{

namespace component
{

namespace topology
{

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::HexaID HexaID;

typedef BaseMeshTopology::Hexa Hexa;
typedef BaseMeshTopology::SeqHexas SeqHexas;
typedef BaseMeshTopology::VertexHexas VertexHexas;
typedef BaseMeshTopology::EdgeHexas EdgeHexas;
typedef BaseMeshTopology::QuadHexas QuadHexas;
typedef BaseMeshTopology::HexaEdges HexaEdges;
typedef BaseMeshTopology::HexaQuads HexaQuads;

typedef Hexa Hexahedron;
typedef HexaEdges HexahedronEdges;
typedef HexaQuads HexahedronQuads;



/////////////////////////////////////////////////////////
/// TopologyChange subclasses
/////////////////////////////////////////////////////////



/** indicates that some hexahedra were added */
class HexahedraAdded : public core::componentmodel::topology::TopologyChange
{

public:
    unsigned int nHexahedra;

    //protected:
    sofa::helper::vector< Hexahedron > hexahedronArray;

    sofa::helper::vector< unsigned int > hexahedronIndexArray;

public:
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;

    sofa::helper::vector< sofa::helper::vector< double > > coefs;

public:

    HexahedraAdded(const unsigned int nT,
            const sofa::helper::vector< Hexahedron >& _hexahedronArray = (const sofa::helper::vector< Hexahedron >)0,
            const sofa::helper::vector< unsigned int >& hexahedraIndex = (const sofa::helper::vector< unsigned int >)0,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors = (const sofa::helper::vector< sofa::helper::vector< unsigned int > >)0,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs = (const sofa::helper::vector< sofa::helper::vector< double > >)0)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::HEXAHEDRAADDED), nHexahedra(nT), hexahedronArray(_hexahedronArray), hexahedronIndexArray(hexahedraIndex),ancestorsList(ancestors), coefs(baryCoefs)
    {   }

    unsigned int getNbAddedHexahedra() const
    {
        return nHexahedra;
    }

};



/** indicates that some hexahedra are about to be removed */
class HexahedraRemoved : public core::componentmodel::topology::TopologyChange
{

    //protected:
public:
    sofa::helper::vector<unsigned int> removedHexahedraArray;



public:

    HexahedraRemoved(const sofa::helper::vector<unsigned int> _tArray) : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::HEXAHEDRAREMOVED), removedHexahedraArray(_tArray)
    {
    }

    const sofa::helper::vector<unsigned int> &getArray() const
    {
        return removedHexahedraArray;
    }

    unsigned int getNbRemovedHexahedra() const
    {
        return removedHexahedraArray.size();
    }

};



/////////////////////////////////////////////////////////
/// HexahedronSetTopology objects
/////////////////////////////////////////////////////////


/** a class that stores a set of hexahedra and provides access with adjacent quads, edges and vertices */
class HexahedronSetTopologyContainer : public QuadSetTopologyContainer
{
private:
    /** \brief Creates the array of edge indices for each hexahedron
    *
    * This function is only called if the HexahedronEdge array is required.
    * m_hexahedronEdge[i] contains the 12 indices of the 12 edges of each hexahedron
    */
    void createHexahedronEdgeArray();
    /** \brief Creates the array of quad indices for each hexahedron
    *
    * This function is only called if the HexahedronQuad array is required.
    * m_hexahedronQuad[i] contains the 6 indices of the 6 quads opposite to the ith vertex
    */
    void createHexahedronQuadArray();
    /** \brief Creates the Hexahedron Vertex Shell Array
    *
    * This function is only called if the HexahedronVertexShell array is required.
    * m_hexahedronVertexShell[i] contains the indices of all hexahedra adjacent to the ith vertex
    */
    void createHexahedronVertexShellArray();

    /** \brief Creates the Hexahedron Edge Shell Array
    *
    * This function is only called if the HexahedronEdheShell array is required.
    * m_hexahedronEdgeShell[i] contains the indices of all hexahedra adjacent to the ith edge
    */
    void createHexahedronEdgeShellArray();
    /** \brief Creates the Hexahedron Quad Shell Array
    *
    * This function is only called if the HexahedronQuadShell array is required.
    * m_hexahedronQuadShell[i] contains the indices of all hexahedra adjacent to the ith edge
    */
    void createHexahedronQuadShellArray();
protected:
    /// provides the set of hexahedra
    sofa::helper::vector<Hexahedron> m_hexahedron;
    /// provides the set of edges for each hexahedron
    sofa::helper::vector<HexahedronEdges> m_hexahedronEdge;
    /// provides the set of quads for each hexahedron
    sofa::helper::vector<HexahedronQuads> m_hexahedronQuad;

    /// for each vertex provides the set of hexahedra adjacent to that vertex
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_hexahedronVertexShell;
    /// for each edge provides the set of hexahedra adjacent to that edge
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_hexahedronEdgeShell;
    /// for each quad provides the set of hexahedra adjacent to that edge
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_hexahedronQuadShell;


    /** \brief Creates the EdgeSet array.
     *
     * Create the set of edges when needed.
    */
    virtual void createEdgeSetArray() {createHexahedronEdgeArray();}

    /** \brief Creates the QuadSet array.
     *
     * Create the array of quads
    */
    virtual void createQuadSetArray() {createHexahedronQuadArray();}

    /** \brief Creates the HexahedronSet array.
     *
     * This function is only called by derived classes to create a list of edges from a set of hexahedra or hexahedra
    */
    virtual void createHexahedronSetArray() {}

public:

    inline friend std::ostream& operator<< (std::ostream& out, const HexahedronSetTopologyContainer& t)
    {
        out  << t.m_hexahedron<< " "
                << t.m_hexahedronEdge<< " "
                << t.m_hexahedronQuad;

        out << " "<< t.m_hexahedronVertexShell.size();
        for (unsigned int i=0; i<t.m_hexahedronVertexShell.size(); i++)
        {
            out << " " << t.m_hexahedronVertexShell[i];
        }
        out <<" "<< t.m_hexahedronEdgeShell.size();
        for (unsigned int i=0; i<t.m_hexahedronEdgeShell.size(); i++)
        {
            out << " " << t.m_hexahedronEdgeShell[i];
        }
        out <<" "<< t.m_hexahedronQuadShell.size();
        for (unsigned int i=0; i<t.m_hexahedronQuadShell.size(); i++)
        {
            out << " " << t.m_hexahedronQuadShell[i];
        }
        return out;
    }

    inline friend std::istream& operator>>(std::istream& in, HexahedronSetTopologyContainer& t)
    {
        unsigned int s;
        sofa::helper::vector< unsigned int > value;


        in >> t.m_hexahedron >> t.m_hexahedronEdge >> t.m_hexahedronQuad;


        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> value;
            t.m_hexahedronVertexShell.push_back(value);
        }
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> value;
            t.m_hexahedronEdgeShell.push_back(value);
        }
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> value;
            t.m_hexahedronQuadShell.push_back(value);
        }
        return in;
    }

    /** \brief Returns the Hexahedron array.
     *
     */
    const sofa::helper::vector<Hexahedron> &getHexahedronArray();

    /** \brief Returns the ith Hexahedron.
     *
     */
    const Hexahedron &getHexahedron(const unsigned int i);

    /** \brief Returns the number of hexahedra in this topology.
     *
     */
    unsigned int getNumberOfHexahedra() ;

    /** \brief Returns the Hexahedron Vertex Shells array.
     *
     */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getHexahedronVertexShellArray() ;

    /** \brief Returns the set of hexahedra adjacent to a given vertex.
     *
     */
    const sofa::helper::vector< unsigned int > &getHexahedronVertexShell(const unsigned int i) ;

    /** \brief Returns the Hexahedron Edges  array.
     *
     */
    const sofa::helper::vector< HexahedronEdges > &getHexahedronEdgeArray() ;

    /** \brief Returns the 12 edges adjacent to a given hexahedron.
     *
     */
    const HexahedronEdges &getHexahedronEdges(const unsigned int i) ;

    /** \brief Returns for each index (between 0 and 12) the two vertex indices that are adjacent to that edge
     *
     */
    Edge getLocalHexahedronEdges (const unsigned int i) const;

    /** \brief Returns the Hexahedron Quads  array.
     *
     */
    const sofa::helper::vector< HexahedronQuads > &getHexahedronQuadArray() ;

    /** \brief Returns the 6 quads adjacent to a given hexahedron.
     *
     */
    const HexahedronQuads &getHexahedronQuads(const unsigned int i) ;

    /** \brief Returns the Hexahedron Edge Shells array.
     *
     */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getHexahedronEdgeShellArray() ;

    /** \brief Returns the set of hexahedra adjacent to a given edge.
     *
     */
    const sofa::helper::vector< unsigned int > &getHexahedronEdgeShell(const unsigned int i) ;


    /** \brief Returns the Hexahedron Quad Shells array.
     *
     */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getHexahedronQuadShellArray() ;

    /** \brief Returns the set of hexahedra adjacent to a given quad.
     *
     */
    const sofa::helper::vector< unsigned int > &getHexahedronQuadShell(const unsigned int i) ;


    /** Returns the indices of a hexahedron given 8 vertex indices : returns -1 if none */
    int getHexahedronIndex(const unsigned int v1, const unsigned int v2, const unsigned int v3, const unsigned int v4, const unsigned int v5, const unsigned int v6, const unsigned int v7, const unsigned int v8);


    /** returns the index of the vertex whose global index is vertexIndex. Returns -1 if none */
    int getVertexIndexInHexahedron(Hexahedron &t,unsigned int vertexIndex) const;

    /** returns the index of the edge whose global index is edgeIndex. Returns -1 if none */
    int getEdgeIndexInHexahedron(const HexahedronEdges &t,unsigned int edgeIndex) const;

    /** returns the index of the quad whose global index is quadIndex. Returns -1 if none */
    int getQuadIndexInHexahedron(const HexahedronQuads &t,unsigned int quadIndex) const;


    /** \brief Checks if the Hexahedron Set Topology is coherent
     *
     */
    virtual bool checkTopology() const;


    HexahedronSetTopologyContainer(core::componentmodel::topology::BaseTopology *top=NULL,
            const sofa::helper::vector< Hexahedron >         &hexahedra    = (const sofa::helper::vector< Hexahedron >)        0 );

    template< typename DataTypes >
    friend class HexahedronSetTopologyModifier;
protected:
    /** \brief Returns a non-const hexahedron vertex shell given a vertex index for subsequent modification
     *
     */
    sofa::helper::vector< unsigned int > &getHexahedronVertexShellForModification(const unsigned int vertexIndex);
    /** \brief Returns a non-const hexahedron edge shell given the index of an edge for subsequent modification
      *
      */
    sofa::helper::vector< unsigned int > &getHexahedronEdgeShellForModification(const unsigned int edgeIndex);


};


template <class DataTypes>
class HexahedronSetTopologyLoader;

/**
 * A class that modifies the topology by adding and removing hexahedra
 */
template<class DataTypes>
class HexahedronSetTopologyModifier : public QuadSetTopologyModifier <DataTypes>
{

public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    HexahedronSetTopologyModifier(core::componentmodel::topology::BaseTopology *top) : QuadSetTopologyModifier<DataTypes>(top)
    {
    }
    /** \brief Build  a hexahedron set topology from a file : also modifies the MechanicalObject
     *
     */
    virtual bool load(const char *filename);


    /** \brief Sends a message to warn that some hexahedra were added in this topology.
     *
     * \sa addHexahedraProcess
     */
    void addHexahedraWarning(const unsigned int nHexahedra,
            const sofa::helper::vector< Hexahedron >& hexahedraList,
            const sofa::helper::vector< unsigned int >& hexahedraIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors= (const sofa::helper::vector< sofa::helper::vector<unsigned int > >) 0 ,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs= (const sofa::helper::vector< sofa::helper::vector< double > >)0) ;



    /** \brief Actually Add some hexahedra to this topology.
     *
     * \sa addHexahedraWarning
     */
    virtual void addHexahedraProcess(const sofa::helper::vector< Hexahedron > &hexahedra);

    /** \brief Sends a message to warn that some hexahedra are about to be deleted.
     *
     * \sa removeHexahedraProcess
     *
     * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
     */
    void removeHexahedraWarning( sofa::helper::vector<unsigned int> &hexahedra);

    /** \brief Remove a subset of hexahedra
     *
     * Elements corresponding to these points are removed form the mechanical object's state vectors.
     *
     * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
     * \sa removeHexahedraWarning
     * @param removeIsolatedItems if true remove isolated quads, edges and vertices
     */
    virtual void removeHexahedraProcess( const sofa::helper::vector<unsigned int> &indices,const bool removeIsolatedItems=false);

    /** \brief Actually Add some quads to this topology.
       *
       * \sa addQuadsWarning
       */
    virtual void addQuadsProcess(const sofa::helper::vector< Quad > &quads);


    /** \brief Remove a subset of quads
     *
     * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
     * @param removeIsolatedEdges if true isolated edges are also removed
     * @param removeIsolatedPoints if true isolated vertices are also removed
     */
    virtual void removeQuadsProcess(const sofa::helper::vector<unsigned int> &indices, const bool removeIsolatedEdges=false, const bool removeIsolatedPoints=false);

    /** \brief Add some edges to this topology.
    *
    * \sa addEdgesWarning
    */
    virtual void addEdgesProcess(const sofa::helper::vector< Edge > &edges);


    /** \brief Remove a subset of edges
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeEdgesWarning
    *
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    * @param removeIsolatedItems if true remove isolated vertices
    */
    virtual void removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,const bool removeIsolatedItems=false);



    /** \brief Add some points to this topology.
     *
     * Use a list of ancestors to create the new points.
     * Last parameter baryCoefs defines the coefficient used for the creation of the new points.
     * Default value for these coefficient (when none is defined) is 1/n with n being the number of ancestors
     * for the point being created.
     *
     * \sa addPointsWarning
     */
    virtual void addPointsProcess(const unsigned int nPoints,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors = (const sofa::helper::vector< sofa::helper::vector< unsigned int > >)0,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs = (const sofa::helper::vector< sofa::helper::vector< double > >)0 );

    /** \brief Add a new point (who has no ancestors) to this topology.
     *
     * \sa addPointsWarning
     */
    virtual void addNewPoint( const sofa::helper::vector< double >& x) {QuadSetTopologyModifier< DataTypes >::addNewPoint(x);};

    /** \brief Remove a subset of points
     *
     * Elements corresponding to these points are removed form the mechanical object's state vectors.
     *
     * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removePointsWarning before calling removePointsProcess.
     * \sa removePointsWarning
     * Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
     */
    virtual void removePointsProcess( sofa::helper::vector<unsigned int> &indices, const bool removeDOF = true);


    /** \brief Reorder this topology.
     *
     * Important : the points are actually renumbered in the mechanical object's state vectors iff (renumberDOF == true)
     * \see MechanicalObject::renumberValues
     */
    virtual void renumberPointsProcess( const sofa::helper::vector<unsigned int> &index, const sofa::helper::vector<unsigned int> &/*inv_index*/, const bool renumberDOF = true);


protected:
    /** \brief Load a hexahedron.
    */
    void addHexahedron(Hexahedron e);

public:
    //template <class DataTypes>
    friend class HexahedronSetTopologyLoader<DataTypes>;

};



/**
 * A class that performs topology algorithms on an HexahedronSet.
 */
template < class DataTypes >
class HexahedronSetTopologyAlgorithms : public QuadSetTopologyAlgorithms<DataTypes>
{

public:

    typedef typename DataTypes::Real Real;

    HexahedronSetTopologyAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top) : QuadSetTopologyAlgorithms<DataTypes>(top)
    {
    }

    /** \brief Remove a set  of hexahedra
    	@param hexahedra an array of hexahedron indices to be removed (note that the array is not const since it needs to be sorted)
    	*
    	*/
    virtual void removeHexahedra(sofa::helper::vector< unsigned int >& hexahedra);

    /** \brief Generic method to remove a list of items.
    */
    virtual void removeItems(sofa::helper::vector< unsigned int >& items);

    /** \brief Generic method for points renumbering
    */
    virtual void renumberPoints( const sofa::helper::vector<unsigned int> &/*index*/, const sofa::helper::vector<unsigned int> &/*inv_index*/);



};

/**
 * A class that provides geometry information on an HexahedronSet.
 */
template < class DataTypes >
class HexahedronSetGeometryAlgorithms : public EdgeSetGeometryAlgorithms<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;


    HexahedronSetGeometryAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top) : EdgeSetGeometryAlgorithms<DataTypes>(top)
    {
    }

    /// computes the volume of hexahedron no i and returns it
    Real computeHexahedronVolume(const unsigned int i) const;
    /// computes the hexahedron volume of all hexahedra are store in the array interface
    void computeHexahedronVolume( BasicArrayInterface<Real> &ai) const;
    /// computes the hexahedron volume  of hexahedron no i and returns it
    Real computeRestHexahedronVolume(const unsigned int i) const;
};


/** Describes a topological object that only consists as a set of points :
it is a base class for all topological objects */
template<class DataTypes>
class HexahedronSetTopology : public QuadSetTopology <DataTypes>
{

public:
    HexahedronSetTopology(component::MechanicalObject<DataTypes> *obj);
    DataPtr<HexahedronSetTopologyContainer > *f_m_topologyContainer;

    virtual void init();
    /** \brief Returns the HexahedronSetTopologyContainer object of this HexahedronSetTopology.
     */
    HexahedronSetTopologyContainer *getHexahedronSetTopologyContainer() const
    {
        return (HexahedronSetTopologyContainer *)this->m_topologyContainer;
    }
    /** \brief Returns the HexahedronSetTopologyAlgorithms object of this HexahedronSetTopology.
     */
    HexahedronSetTopologyAlgorithms<DataTypes> *getHexahedronSetTopologyAlgorithms() const
    {
        return (HexahedronSetTopologyAlgorithms<DataTypes> *)this->m_topologyAlgorithms;
    }

    /** \brief Returns the TriangleSetTopologyAlgorithms object of this TriangleSetTopology.
    */
    virtual core::componentmodel::topology::TopologyAlgorithms *getTopologyAlgorithms() const
    {
        return getHexahedronSetTopologyAlgorithms();
    }

    /** \brief Returns the HexahedronSetTopologyAlgorithms object of this HexahedronSetTopology.
     */
    HexahedronSetGeometryAlgorithms<DataTypes> *getHexahedronSetGeometryAlgorithms() const
    {
        return (HexahedronSetGeometryAlgorithms<DataTypes> *)this->m_geometryAlgorithms;
    }

    /// BaseMeshTopology API
    /// @{

    const SeqHexas& getHexas()   { return getHexahedronSetTopologyContainer()->getHexahedronArray(); }
    /// Returns the set of edges adjacent to a given hexahedron.
    const HexaEdges& getEdgeHexaShell(HexaID i) { return getHexahedronSetTopologyContainer()->getHexahedronEdges(i); }
    /// Returns the set of quads adjacent to a given hexahedron.
    const HexaQuads& getQuadHexaShell(HexaID i) { return getHexahedronSetTopologyContainer()->getHexahedronQuads(i); }
    /// Returns the set of hexahedra adjacent to a given vertex.
    const VertexHexas& getHexaVertexShell(PointID i) { return getHexahedronSetTopologyContainer()->getHexahedronVertexShell(i); }
    /// Returns the set of hexahedra adjacent to a given edge.
    const EdgeHexas& getHexaEdgeShell(EdgeID i) { return getHexahedronSetTopologyContainer()->getHexahedronEdgeShell(i); }
    /// Returns the set of hexahedra adjacent to a given quad.
    const QuadHexas& getHexaQuadShell(QuadID i) { return getHexahedronSetTopologyContainer()->getHexahedronQuadShell(i); }

    /// @}

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
