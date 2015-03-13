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

#ifndef __IMPLICIT_HIERARCHICAL_MAP3__
#define __IMPLICIT_HIERARCHICAL_MAP3__

#include "Topology/map/embeddedMap3.h"

namespace CGoGN
{

template<typename T, unsigned int ORBIT> class AttributeHandler_IHM ;



class ImplicitHierarchicalMap3 : public EmbeddedMap3
{
    template<typename T, unsigned int ORBIT> friend class AttributeHandler_IHM ;
	typedef EmbeddedMap3::TOPO_MAP TOPO_MAP;

private:
    unsigned int m_curLevel ;
    unsigned int m_maxLevel ;
    unsigned int m_edgeIdCount ;
    unsigned int m_faceIdCount;

    DartAttribute<unsigned int, ImplicitHierarchicalMap3> m_dartLevel ;
    DartAttribute<unsigned int, ImplicitHierarchicalMap3> m_edgeId ;
    DartAttribute<unsigned int, ImplicitHierarchicalMap3> m_faceId ;

    AttributeMultiVector<unsigned int>* m_nextLevelCell[NB_ORBITS] ;

public:
    ImplicitHierarchicalMap3() ;

    ~ImplicitHierarchicalMap3() ;

    static const unsigned int DIMENSION = 3 ;

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

//	template <typename T, unsigned int ORBIT, typename MAP>
//	AttributeHandler_IHM<T, ORBIT, MAP> addAttribute(const std::string& nameAttr) ;

//	template <typename T, unsigned int ORBIT, typename MAP>
//	AttributeHandler_IHM<T, ORBIT, MAP> getAttribute(const std::string& nameAttr) ;

    /***************************************************
     *                 MAP TRAVERSAL                   *
     ***************************************************/

    inline Dart newDart() ;

    inline Dart phi1(Dart d) const;

    inline Dart phi_1(Dart d) const;

    inline Dart phi2(Dart d) const;

private:
    inline Dart phi2bis(Dart d) const;

public:
    inline Dart phi3(Dart d) const;

    inline Dart alpha0(Dart d) const;

    inline Dart alpha1(Dart d) const;

    inline Dart alpha2(Dart d) const;

    inline Dart alpha_2(Dart d) const;

    inline Dart begin() const;

    inline Dart end() const;

    inline void next(Dart& d) const ;

    template <unsigned int ORBIT, typename FUNC>
    void foreach_dart_of_orbit(Cell<ORBIT> c, const FUNC& f) const ;
//    template <unsigned int ORBIT, typename FUNC>
//	void foreach_dart_of_orbit(Cell<ORBIT> c, FUNC& f) const ;

    template <typename FUNC>
    void foreach_dart_of_vertex(Dart d, const FUNC& f) const ;

    template <typename FUNC>
    void foreach_dart_of_edge(Dart d, const FUNC& f) const ;

    template <typename FUNC>
    void foreach_dart_of_oriented_face(Dart d, const FUNC& f) const;

    template <typename FUNC>
    void foreach_dart_of_face(Dart d, const FUNC& f) const ;

    template <typename FUNC>
    void foreach_dart_of_oriented_volume(Dart d, const FUNC& f) const;

    template <typename FUNC>
    void foreach_dart_of_volume(Dart d, const FUNC& f) const ;

    template <typename FUNC>
    void foreach_dart_of_vertex1(Dart d, const FUNC& f) const;

    template <typename FUNC>
    void foreach_dart_of_edge1(Dart d, const FUNC& f) const;

    template <typename FUNC>
    void foreach_dart_of_vertex2(Dart d, const FUNC& f) const;

    template <typename FUNC>
    void foreach_dart_of_edge2(Dart d, const FUNC& f) const;

    template <typename FUNC>
    void foreach_dart_of_face2(Dart d, const FUNC& f) const;

    template <typename FUNC>
    void foreach_dart_of_cc(Dart d, const FUNC& f) const ;

    /***************************************************
     *               MAP MANIPULATION                  *
     ***************************************************/


    /***************************************************
     *              LEVELS MANAGEMENT                  *
     ***************************************************/
    void incCurrentLevel();

    void decCurrentLevel();

    unsigned int getCurrentLevel() const ;

    void setCurrentLevel(unsigned int l) ;

    unsigned int getMaxLevel() const ;

    unsigned int getDartLevel(Dart d) const ;

    void setDartLevel(Dart d, unsigned int i) ;

    /***************************************************
     *                  ID MANAGEMENT                  *
     ***************************************************/
    //! Give a new unique id to all the edges of the map
    /*!
     */
    void initEdgeId() ;

    //! Return the next available edge id
    /*!
     */
    unsigned int getNewEdgeId() ;

    //! Return the id of the edge of d
    /*!
     */
    unsigned int getEdgeId(Dart d) ;

    //! Set an edge id to all darts from an orbit of d
    /*!
     */
	void setEdgeId(Dart d, unsigned int i); //TODO a virer
	void setDartEdgeId(Dart d, unsigned int i);

	unsigned int triRefinementEdgeId(Dart d);

	unsigned int quadRefinementEdgeId(Dart d);




    //! Give a new unique id to all the faces of the map
    /*!
     */
    void initFaceId() ;

    //! Return the next available face id
    /*!
     */
    unsigned int getNewFaceId() ;

    //! Return the id of the face of d
    /*!
     */
    unsigned int getFaceId(Dart d) ;

	unsigned int faceId(Dart d);

    //! Set a face id to all darts from an orbit of d
    /*!
     */
    void setFaceId(Dart d, unsigned int i, unsigned int orbit); //TODO a virer
    void setFaceId(unsigned int orbit, Dart d);

    /***************************************************
     *               CELLS INFORMATION                 *
     ***************************************************/

    //! Return the level of insertion of the vertex of d
    /*!
     */
    unsigned int vertexInsertionLevel(Dart d) const;
};

//TODO existe deja dans le fichier ihm2.h
//template <typename T, unsigned int ORBIT>
//class AttributeHandler_IHM : public AttributeHandler<T, ORBIT, ImplicitHierarchicalMap3>
//{
//public:
//    typedef T DATA_TYPE ;

//    AttributeHandler_IHM() : AttributeHandler<T, ORBIT, ImplicitHierarchicalMap3>()
//    {}

//    AttributeHandler_IHM(ImplicitHierarchicalMap3* m, AttributeMultiVector<T>* amv) : AttributeHandler<T, ORBIT, ImplicitHierarchicalMap3>(m, amv)
//    {}

//    AttributeMultiVector<T>* getDataVector() const
//    {
//        return AttributeHandler<T, ORBIT, ImplicitHierarchicalMap3>::getDataVector() ;
//    }

//    bool isValid() const
//    {
//        return AttributeHandler<T, ORBIT, ImplicitHierarchicalMap3>::isValid() ;
//    }

//    virtual T& operator[](Dart d) ;

//    virtual const T& operator[](Dart d) const ;

//    T& operator[](unsigned int a)
//    {
//        return AttributeHandler<T, ORBIT, ImplicitHierarchicalMap3>::operator[](a) ;
//    }

//    const T& operator[](unsigned int a) const
//    {
//        return AttributeHandler<T, ORBIT, ImplicitHierarchicalMap3>::operator[](a) ;
//    }

//} ;

//template <typename T>
//class VertexAttribute_IHM : public AttributeHandler_IHM<T, VERTEX>
//{
//public:
//    VertexAttribute_IHM() : IHM::AttributeHandler_IHM<T, VERTEX>() {}
//    VertexAttribute_IHM(const IHM::AttributeHandler_IHM<T, VERTEX>& ah) : IHM::AttributeHandler_IHM<T, VERTEX>(ah) {}
////	VertexAttribute_IHM<T>& operator=(const IHM::AttributeHandler_IHM<T, VERTEX>& ah) { this->IHM::AttributeHandler_IHM<T, VERTEX>::operator=(ah); return *this; }
//};


} //namespace CGoGN

#include "Topology/ihmap/ihm3.hpp"

#endif
