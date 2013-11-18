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

#ifndef __IMPLICIT_HIERARCHICAL_MAP__
#define __IMPLICIT_HIERARCHICAL_MAP__

#include "Topology/map/embeddedMap2.h"

namespace CGoGN
{

namespace Algo
{
namespace Surface
{
namespace IHM
{

template<typename T, unsigned int ORBIT> class AttributeHandler_IHM ;

class ImplicitHierarchicalMap : public EmbeddedMap2
{
	template<typename T, unsigned int ORBIT> friend class AttributeHandler_IHM ;

private:
	unsigned int m_curLevel ;
	unsigned int m_maxLevel ;
	unsigned int m_idCount ;

	DartAttribute<unsigned int> m_dartLevel ;
	DartAttribute<unsigned int> m_edgeId ;

	AttributeMultiVector<unsigned int>* m_nextLevelCell[NB_ORBITS] ;

public:
	ImplicitHierarchicalMap() ;

	~ImplicitHierarchicalMap() ;

	static const unsigned int DIMENSION = 2 ;

	//!
	/*!
	 *
	 */
	void update_topo_shortcuts();


	void initImplicitProperties() ;

	/**
	 * clear the map
	 * @param remove attrib remove attribute (not only clear the content)
	 */
	void clear(bool removeAttrib);


	/***************************************************
	 *             ATTRIBUTES MANAGEMENT               *
	 ***************************************************/

	template <typename T, unsigned int ORBIT>
	AttributeHandler_IHM<T, ORBIT> addAttribute(const std::string& nameAttr) ;

	template <typename T, unsigned int ORBIT>
	AttributeHandler_IHM<T, ORBIT> getAttribute(const std::string& nameAttr) ;

	/***************************************************
	 *                 MAP TRAVERSAL                   *
	 ***************************************************/

	virtual Dart newDart() ;

	Dart phi1(Dart d) ;

	Dart phi_1(Dart d) ;

	Dart phi2(Dart d) ;

	Dart alpha0(Dart d) ;

	Dart alpha1(Dart d) ;

	Dart alpha_1(Dart d) ;

	virtual Dart begin() const ;

	virtual Dart end() const ;

	virtual void next(Dart& d) const ;

	virtual bool foreach_dart_of_vertex(Dart d, FunctorType& f, unsigned int thread = 0) ;

	virtual bool foreach_dart_of_edge(Dart d, FunctorType& f, unsigned int thread = 0) ;

	virtual bool foreach_dart_of_oriented_face(Dart d, FunctorType& f, unsigned int thread = 0) ;
	virtual bool foreach_dart_of_face(Dart d, FunctorType& f, unsigned int thread = 0) ;

	virtual bool foreach_dart_of_oriented_volume(Dart d, FunctorType& f, unsigned int thread = 0) ;
	virtual bool foreach_dart_of_volume(Dart d, FunctorType& f, unsigned int thread = 0) ;

	virtual bool foreach_dart_of_cc(Dart d, FunctorType& f, unsigned int thread = 0) ;

	/***************************************************
	 *               MAP MANIPULATION                  *
	 ***************************************************/

	void splitFace(Dart d, Dart e) ;

	/***************************************************
	 *              LEVELS MANAGEMENT                  *
	 ***************************************************/

	unsigned int getCurrentLevel() ;

	void setCurrentLevel(unsigned int l) ;

	unsigned int getMaxLevel() ;

	unsigned int getDartLevel(Dart d) ;

	void setDartLevel(Dart d, unsigned int i) ;


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

	unsigned int getMaxEdgeId();

	/***************************************************
	 *               CELLS INFORMATION                 *
	 ***************************************************/

	/**
	 * Return the level of insertion of the vertex of d
	 */
	unsigned int vertexInsertionLevel(Dart d) ;

	/**
	 * Return the level of the edge of d in the current level map
	 */
	unsigned int edgeLevel(Dart d) ;

	/**
	 * Return the level of the face of d in the current level map
	 */
	unsigned int faceLevel(Dart d) ;

	/**
	 * Given the face of d in the current level map,
	 * return a level 0 dart of its origin face
	 */
	Dart faceOrigin(Dart d) ;

	/**
	 * Return the oldest dart of the face of d in the current level map
	 */
	Dart faceOldestDart(Dart d) ;

	//! Test if dart d and e belong to the same face
	/*! @param d a dart
	 *  @param e a dart
	 */
	bool sameFace(Dart d, Dart e) ;

	/**
	 * Return true if the edge of d in the current level map
	 * has already been subdivided to the next level
	 */
	bool edgeIsSubdivided(Dart d) ;

	/**
	 * Return true if the edge of d in the current level map
	 * is subdivided to the next level,
	 * none of its resulting edges is in turn subdivided to the next level
	 * and the middle vertex is of degree 2
	 */
	bool edgeCanBeCoarsened(Dart d) ;

	/**
	 * Return true if the face of d in the current level map
	 * has already been subdivided to the next level
	 */
	bool faceIsSubdivided(Dart d) ;

	/**
	 * Return true if the face of d in the current level map
	 * is subdivided to the next level
	 * and none of its resulting faces is in turn subdivided to the next level
	 */
	bool faceIsSubdividedOnce(Dart d) ;
} ;

template <typename T, unsigned int ORBIT>
class AttributeHandler_IHM : public AttributeHandler<T, ORBIT>
{
public:
	typedef T DATA_TYPE ;

	AttributeHandler_IHM() : AttributeHandler<T, ORBIT>()
	{}

	AttributeHandler_IHM(GenericMap* m, AttributeMultiVector<T>* amv) : AttributeHandler<T, ORBIT>(m, amv)
	{}

	AttributeMultiVector<T>* getDataVector() const
	{
		return AttributeHandler<T, ORBIT>::getDataVector() ;
	}

	bool isValid() const
	{
		return AttributeHandler<T, ORBIT>::isValid() ;
	}

	T& operator[](Dart d) ;

	const T& operator[](Dart d) const ;

	T& operator[](unsigned int a)
	{
		return AttributeHandler<T, ORBIT>::operator[](a) ;
	}

	const T& operator[](unsigned int a) const
	{
		return AttributeHandler<T, ORBIT>::operator[](a) ;
	}
} ;

} //namespace IHM
} // Surface
} //namespace Algo

} //namespace CGoGN

#include "Algo/ImplicitHierarchicalMesh/ihm.hpp"

#endif
