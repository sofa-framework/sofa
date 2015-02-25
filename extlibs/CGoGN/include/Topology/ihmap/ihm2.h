/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
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
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#ifndef __IMPLICIT_HIERARCHICAL_MAP2__
#define __IMPLICIT_HIERARCHICAL_MAP2__

#include "Topology/map/embeddedMap2.h"
#include "Topology/generic/mapImpl/mapIH2.h"


namespace CGoGN
{

class ImplicitHierarchicalMap2 : public EmbeddedMap2
{ 
    template <typename T, unsigned int ORBIT, class MAP>
    friend class AttributeHandler;
private:
    unsigned int m_curLevel ;
    unsigned int m_maxLevel ;
    unsigned int m_idCount ;

    DartAttribute<unsigned int, EmbeddedMap2> m_dartLevel ;
    DartAttribute<unsigned int, EmbeddedMap2> m_edgeId ;

    AttributeMultiVector<unsigned int>* m_nextLevelCell[NB_ORBITS] ;

public:
    ImplicitHierarchicalMap2() ;
    ~ImplicitHierarchicalMap2() ;

    //!
    /*!
     *
     */
    void update_topo_shortcuts();

    //!
    /*!
     *
     */
    void initImplicitProperties();

    /**
     * clear the map
     * @param remove attrib remove attribute (not only clear the content)
     */
    void clear(bool removeAttrib);

    /***************************************************
     *             ATTRIBUTES MANAGEMENT               *
     ***************************************************/

    template <typename T, unsigned int ORBIT, typename MAP>
    AttributeHandler<T, ORBIT, MAP> addAttribute(const std::string& nameAttr) ;

    template <typename T, unsigned int ORBIT, typename MAP>
    inline AttributeHandler<T, ORBIT, MAP> getAttribute(const std::string& nameAttr) ;

    /***************************************************
     *                 MAP TRAVERSAL                   *
     ***************************************************/

    inline Dart newDart() ;

    inline Dart phi1(Dart d) const ;

    inline Dart phi_1(Dart d) const ;

    inline Dart phi2(Dart d) const ;

    inline Dart alpha0(Dart d) const ;

    inline Dart alpha1(Dart d) const ;

    inline Dart alpha_1(Dart d) const ;

    inline Dart begin() const ;

    inline Dart end() const ;

    inline void next(Dart& d) const ;

    //	template <unsigned int ORBIT, typename FUNC>
    //	void foreach_dart_of_orbit(Cell<ORBIT> c, FUNC f) const ;
    template <unsigned int ORBIT, typename FUNC>
    void foreach_dart_of_orbit(Cell<ORBIT> c, const FUNC& f) const ;

    template <typename FUNC>
    void foreach_dart_of_vertex(Dart d, const FUNC& f) const;

    template <typename FUNC>
    void foreach_dart_of_edge(Dart d, const FUNC& f) const ;

    template <typename FUNC>
    void foreach_dart_of_oriented_face(Dart d, const FUNC& f) const ;

    template <typename FUNC>
    void foreach_dart_of_face(Dart d, const FUNC& f)  const;

    template <typename FUNC>
    void foreach_dart_of_oriented_volume(Dart d, const FUNC& f) const ;

    template <typename FUNC>
    void foreach_dart_of_volume(Dart d, const FUNC& f) const ;

    template <typename FUNC>
    void foreach_dart_of_vertex1(Dart d, const FUNC& f) const;

    template <typename FUNC>
    void foreach_dart_of_edge1(Dart d, const FUNC& f) const;

    template <typename FUNC>
    void foreach_dart_of_cc(Dart d, const FUNC& f) const ;

    /***************************************************
     *               MAP MANIPULATION                  *
     ***************************************************/

    void splitFace(Dart d, Dart e) ;

    unsigned int vertexDegree(Dart d);

    /***************************************************
     *              LEVELS MANAGEMENT                  *
     ***************************************************/

    unsigned int getCurrentLevel() ;

    void setCurrentLevel(unsigned int l) ;

    void incCurrentLevel();

    void decCurrentLevel();

    unsigned int getMaxLevel() ;

    unsigned int getDartLevel(Dart d) ;

    void setDartLevel(Dart d, unsigned int i) ;

    void setMaxLevel(unsigned int l);

    /***************************************************
     *             EDGE ID MANAGEMENT                  *
     ***************************************************/

    /**
     * Give a new unique id to all the edges of the map
     */
    void initEdgeId() ;

    /**
     * Return the next available edge id
     */
    unsigned int getNewEdgeId() ;

    unsigned int getEdgeId(Dart d) ;

    void setEdgeId(Dart d, unsigned int i) ;

    unsigned int getTriRefinementEdgeId(Dart d);

    unsigned int getQuadRefinementEdgeId(Dart d);

    /***************************************************
     *               CELLS INFORMATION                 *
     ***************************************************/

    /**
     * @brief faceDegree
     * @param d
     * @return
     */
    unsigned int faceDegree(Dart d);

    /**
     * Return the level of insertion of the vertex of d
     */
    unsigned int vertexInsertionLevel(Dart d) ;

    //	/**
    //	 * Return the level of the edge of d in the current level map
    //	 */
    //	unsigned int edgeLevel(Dart d) ;

    //	/**
    //	 * Return the level of the face of d in the current level map
    //	 */
    //	unsigned int faceLevel(Dart d) ;

    //	/**
    //	 * Given the face of d in the current level map,
    //	 * return a level 0 dart of its origin face
    //	 */
    //	Dart faceOrigin(Dart d) ;

    //	/**
    //	 * Return the oldest dart of the face of d in the current level map
    //	 */
    //	Dart faceOldestDart(Dart d) ;

    //	/**
    //	 * Return true if the edge of d in the current level map
    //	 * has already been subdivided to the next level
    //	 */
    //	bool edgeIsSubdivided(Dart d) ;

    //	/**
    //	 * Return true if the edge of d in the current level map
    //	 * is subdivided to the next level,
    //	 * none of its resulting edges is in turn subdivided to the next level
    //	 * and the middle vertex is of degree 2
    //	 */
    //	bool edgeCanBeCoarsened(Dart d) ;

    //	/**
    //	 * Return true if the face of d in the current level map
    //	 * has already been subdivided to the next level
    //	 */
    //	bool faceIsSubdivided(Dart d) ;

    //	/**
    //	 * Return true if the face of d in the current level map
    //	 * is subdivided to the next level
    //	 * and none of its resulting faces is in turn subdivided to the next level
    //	 */
    //	bool faceIsSubdividedOnce(Dart d) ;
} ;


/**
 * partial specialization for IHM2
 */
template <typename T, unsigned int ORBIT>
class AttributeHandler<T,ORBIT,ImplicitHierarchicalMap2> : public AttributeHandlerGen
{
    friend class ImplicitHierarchicalMap2;
public:
    typedef ImplicitHierarchicalMap2 MAP;
protected:
    MAP* m_map;
    AttributeMultiVector<T>* m_attrib;

    void registerInMap() ;
    void unregisterFromMap() ;
private:
    template <unsigned int ORBIT2>
    AttributeHandler(const AttributeHandler<T, ORBIT2, MAP>& h) ;
    template <unsigned int ORBIT2>
    AttributeHandler<T, ORBIT, MAP>& operator=(const AttributeHandler<T, ORBIT2, MAP>& ta) ;

public:
    typedef T DATA_TYPE ;
    AttributeHandler() ;
    AttributeHandler(MAP* m, AttributeMultiVector<T>* amv) ;
    AttributeHandler(const AttributeHandler<T, ORBIT, MAP>& ta) ;
    AttributeHandler<T, ORBIT, MAP>& operator=(const AttributeHandler<T, ORBIT, MAP>& ta) ;
    virtual ~AttributeHandler() ;
    MAP* map() const
    {
        return m_map ;
    }
    AttributeMultiVector<T>* getDataVector() const ;
    virtual AttributeMultiVectorGen* getDataVectorGen() const ;
    virtual int getSizeOfType() const ;
    virtual unsigned int getOrbit() const ;
    unsigned int getIndex() const ;
    virtual const std::string& name() const ;
    virtual const std::string& typeName() const ;
    unsigned int nbElements() const;
    T& operator[](Cell<ORBIT> c) ;
    const T& operator[](Cell<ORBIT> c) const ;
    T& operator[](unsigned int a) ;
    const T& operator[](unsigned int a) const ;
    unsigned int insert(const T& elt) ;
    unsigned int newElt() ;
    void setAllValues(const T& v) ;
    unsigned int begin() const;
    unsigned int end() const;
    void next(unsigned int& iter) const;
} ;

// TODO : CONTINUE
// NEW VERSION

class IHM2 : public Map2<MapIH2> {
public:
    typedef MapIH2 IMPL;
    typedef Map2<IMPL> TOPO_MAP;
    static const unsigned int DIMENSION = TOPO_MAP::DIMENSION ;

private:
    // Map2 interface
    IHM2(const IHM2& m): TOPO_MAP(m)  {}
public:
     inline IHM2() : TOPO_MAP()
     {}
//    virtual bool checkSimpleOrientedPath(std::vector<Dart> &vd);

    // from ihm2
     inline Dart phi1(Dart d) const ;
     inline Dart phi_1(Dart d) const ;
     inline Dart phi2(Dart d) const ;
     inline Dart alpha0(Dart d) const ;
     inline Dart alpha1(Dart d) const ;
     inline Dart alpha_1(Dart d) const ;

     template <typename T, unsigned int ORBIT, typename MAP>
     AttributeHandler<T, ORBIT, MAP> addAttribute(const std::string& nameAttr) ;
     template <typename T, unsigned int ORBIT, typename MAP>
     inline AttributeHandler<T, ORBIT, MAP> getAttribute(const std::string& nameAttr) ;





     // FROM EMBEDDEDMAP2
    Dart newPolyLine(unsigned int nbEdges) ;
    Dart newFace(unsigned int nbEdges, bool withBoundary = true) ;
    void splitVertex(Dart d, Dart e) ;
    Dart deleteVertex(Dart d) ;
    Dart cutEdge(Dart d) ;
    bool uncutEdge(Dart d) ;
    bool edgeCanCollapse(Dart d) ;
    Dart collapseEdge(Dart d, bool delDegenerateFaces = true) ;
    bool flipEdge(Dart d) ;
    bool flipBackEdge(Dart d) ;
    void swapEdges(Dart d, Dart e);
    void insertEdgeInVertex(Dart d, Dart e);
    bool removeEdgeFromVertex(Dart d);
    void sewFaces(Dart d, Dart e, bool withBoundary = true) ;
    virtual void unsewFaces(Dart d, bool withBoundary = true) ;
    virtual bool collapseDegeneratedFace(Dart d);
    virtual void splitFace(Dart d, Dart e) ;
    bool mergeFaces(Dart d) ;
    bool mergeVolumes(Dart d, Dart e, bool deleteFace = true) ;
    void splitSurface(std::vector<Dart>& vd, bool firstSideClosed = true, bool secondSideClosed = true);
    virtual unsigned int closeHole(Dart d, bool forboundary = true);
    virtual bool check() const;

    /***************************************************
     *             EDGE ID MANAGEMENT                  *
     ***************************************************/

    void initEdgeId() ;
    inline unsigned int getNewEdgeId() ;
    inline unsigned int getEdgeId(Dart d) ;
    inline void setEdgeId(Dart d, unsigned int i) ;
    unsigned int getTriRefinementEdgeId(Dart d);
    unsigned int getQuadRefinementEdgeId(Dart d);

private:
    DartAttribute<unsigned int, TOPO_MAP> m_edgeId ;
};

template <typename T, unsigned int ORBIT, typename MAP>
AttributeHandler<T, ORBIT, MAP> IHM2::addAttribute(const std::string &nameAttr)
{
    bool addNextLevelCell = false ;
    if(!isOrbitEmbedded<ORBIT>())
        addNextLevelCell = true ;

    AttributeHandler<T, ORBIT, MAP> h = TOPO_MAP::addAttribute<T, ORBIT, MAP>(nameAttr) ;

    if(addNextLevelCell)
    {
        AttributeContainer& cellCont = m_attribs[ORBIT] ;
        AttributeMultiVector<unsigned int>* amv = cellCont.addAttribute<unsigned int>("nextLevelCell") ;
        this->m_nextLevelCell[ORBIT] = amv ;
        for(unsigned int i = cellCont.begin(); i < cellCont.end(); cellCont.next(i))
            amv->operator[](i) = EMBNULL ;
    }

    return AttributeHandler<T, ORBIT, MAP>(this, h.getDataVector()) ;
}


template <typename T, unsigned int ORBIT, typename MAP>
inline AttributeHandler<T, ORBIT, MAP> IHM2::getAttribute(const std::string& nameAttr)
{
    return AttributeHandler<T, ORBIT, MAP>(this, TOPO_MAP::getAttribute<T, ORBIT, MAP>(nameAttr).getDataVector()) ;
}













} //namespace CGoGN

#include "Topology/ihmap/ihm2.hpp"

#endif
