/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1																   *
* Copyright (C) 2009, IGG Team, LSIIT, University of Strasbourg				   *
*																			   *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.												   *
*																			   *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or		   *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.															   *
*																			   *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.		   *
*																			   *
* Web site: http://cgogn.unistra.fr/    							   *
* Contact information: cgogn@unistra.fr										   *
*																			   *
*******************************************************************************/

#ifndef __HDART_H__
#define __HDART_H__

#include <iostream>
#include <cassert>

#include "Topology/generic/dartparameters.h"
#include "embedding.h"
#include "Utils/marker.h"
#include "label.h"

namespace CGoGN
{

typedef Emb::Embedding Embedding;

template <typename DART> class tDartList;
template <typename DART> class tGenericMap;
template <typename DART> class tBaseHyperMap;
template <typename DART> class eHyperMap1;


template <typename DP>
class HMDartObj : public Marker, public Label
{
public:
	//! Type definition of a dart iterator.
	typedef typename std::list<HMDartObj>::iterator Dart;

	friend typename tDartList<HMDartObj>::Dart tDartList<HMDartObj>::newDart();
	friend class tGenericMap<HMDartObj>;
	friend class tBaseHyperMap<HMDartObj>;
	friend class eHyperMap1<HMDartObj>;

	//! Constructor
	HMDartObj(unsigned label) : Marker(), Label(label)
	{
		for (unsigned i=0; i < DP::nbEmbeddings; ++i)
			m_emb[i] = NULL;
	}

	//! Destructor
	~HMDartObj()
	{
		for (unsigned i=0; i < DP::nbEmbeddings; ++i)
			this->setEmbedding(i,NULL);
	}

private:
	/*! The relation are stored in the array m_topo[].
	 * Inverse permutations are stored if and only if optimization is enabled.
	 * The cell m_topo[index] contains :
	 *	@param index in [0, nbPermutations[					: permutations
	 *	@param index in [nbPermutations, nbRelations[		: involutions
	 *	@param index in [nbRelations, nbStoredRelations[	: inverse permutations if stored
	 */
	Dart m_topo[2*DP::nbPermutations];

	//! An array that contains the embeddings
	/*! The last cell m_emb[DP::nbEmbeddings] contains a label used as unique identifier
	 */
	Embedding* m_emb[DP::nbEmbeddings];

	/**********************************************************************************************
	 *                           Topological relations management                                 *
	 **********************************************************************************************/

	//! The number of topological relations strored in the dart.
	/*! That includes the inverse of permutations if optimize_perm is true.
	 * @return the size of m_topo
	 */
	static unsigned nbStoredRelations() 
	{
		return 2*DP::nbPermutations;
	}

	//! The number of permutations stored in the dart
	static unsigned nbPermutations() 
	{
		return DP::nbPermutations;
	}

	//! The number of topological relations in the dart. This is also the maximal dimension of the dart.
	/*! @return the number of involutions and permutations
	 */
	static unsigned nbRelations() 
	{
		return DP::nbPermutations;
	}

	/* A set of private methods to manage the combinatorial relation without any topological knowledge.
	 * These methods ensure the relation are true involutions and permutations.
	 */

	//! Get the value of a permutation
	/*!	@param i in [0, nbPermutations[
	 */
	Dart getPermutation(unsigned i) const 
	{
		assert(i<DP::nbPermutations || !"Invalid parameter: i out of bounds");
		return m_topo[i];
	}

	//! Get the value of the inverse of a permutation
	/*!	@param i in [0, nbPermutations[
	 */
	Dart getPermutationInverse(unsigned i) const 
	{
		assert(i<DP::nbPermutations || !"Invalid parameter: i out of bounds");
		return m_topo[nbRelations()+i];
	}

	//! Link the current dart to dart d with a permutation
	static void setPermutation(unsigned i,Dart d, Dart e) 
	{
		assert(i<DP::nbPermutations || !"Invalid parameter: i out of bounds");
		d->m_topo[i] = e;
		e->m_topo[nbRelations()+i] = d;
	}

	//! Unlink the current dart from its successor in a permutation
	void unsetPermutation(unsigned i,Dart nil)
	{
		assert(i<DP::nbPermutations || !"Invalid parameter: i out of bounds");
		Dart d = m_topo[i];
		Dart e = d->m_topo[i];
		m_topo[i] = e;
		d->m_topo[i] = nil;
		Dart c = d->m_topo[nbRelations()+i];
		e->m_topo[nbRelations()+i] = c;
		d->m_topo[nbRelations()+i] = nil;
	}

	/**********************************************************************************************
	 *                             Embedding pointers management                                  *
	 **********************************************************************************************/

	//! The number of embeddings per dart with current parameters
	/*! @return the number of available embeddings
	 */
	static unsigned nbEmbeddings()
	{
		return DP::nbEmbeddings;
	}

	//! Get the i_th embedding
	/*! @param i the index of embedding
	 * 	@return the i_th embedding
	 */
	Embedding* getEmbedding(unsigned i) const
	{
		assert(i < nbEmbeddings() || !"Invalid parameter: i out of bounds");
		return m_emb[i];
	}

	//! Set the i_th embedding
	/*! @param i the index of embedding
	 * 	@param emb the embedding
	 */
	void setEmbedding(unsigned i, Embedding* emb)
	{
		assert(i < nbEmbeddings() || !"Invalid parameter: i out of bounds");
		if (m_emb[i] == emb) return;
		if (m_emb[i] != NULL) Embedding::unref(m_emb[i]);
		if (emb != NULL) Embedding::ref(emb);
		m_emb[i] = emb;
	}

public:
	//! Get vertex embedding index if handled by CGoGN
	static unsigned getVertexEmbId()
	{
		assert(DP::vertexEmb);
		return DP::vertexEmbId;
	}

	//! Get edge embedding index if handled by CGoGN
	static unsigned getEdgeEmbId()
	{
		assert(DP::edgeEmb);
		return DP::edgeEmbId;
	}

	//! Get face embedding index if handled by CGoGN
	static unsigned getFaceEmbId()
	{
		assert(DP::faceEmb);
		return DP::faceEmbId;
	}

	//! Get dart embedding index if handled by CGoGN
	static unsigned getDartEmbId()
	{
		assert(DP::dartEmb);
		return DP::dartEmbId;
	}

private:
	/**********************************************************************************************
	 *                                Input / Output management                                   *
	 **********************************************************************************************/

	//! output topology of dart to stream (use labels)
	friend std::ostream& operator<<(std::ostream& os, const HMDartObj<DP>& d)
	{
		if (nbRelations() > 0)
			os << d.m_topo[0]->getLabel();
		for (unsigned i=1; i < nbRelations(); ++i)
			os << " " << d.m_topo[i]->getLabel();
		return os;
	}

	//! Save labels of relations in a buffer: used by saveMap
	/*! /warning Suppose the map to be correctly reindexed
	 *  Do not use without knowing what it involves
	 */
	unsigned output(unsigned buffer[16])
	{
		unsigned index = 0;

		buffer[index++] = this->getLabel();
		buffer[index++] = this->getMarkerVal();
		for (unsigned i=0; i < DP::nbEmbeddings; ++i) {
			assert(index < 16);
			Embedding* emb = this->getEmbedding(i);
			if (emb != NULL)
				buffer[index++] = emb->getLabel();
			else
				buffer[index++] = Embedding::NO_LABEL();
		}

		for (unsigned i=0; i < nbRelations(); ++i) {
			assert(index < 16);
			buffer[index++] = m_topo[i]->getLabel();
		}
		return index;
	}

	//! Set relations from a vector of Dart: used by loadMap to init a Dart from saved values
	void inputRelations(Dart relations[16]) {
		assert(nbRelations() <= 16);
		unsigned i;
		for (i=0; i < DP::nbPermutations; ++i) {
			if (DP::optimize_perm == true) {
				Dart c = m_topo[i];			// c is an iterator on *this
				assert(c->m_topo[i] == c);	// c should be a fixed point
				relations[i]->m_topo[nbRelations()+i] = c;
			}
			m_topo[i] = relations[i];
		}
		for (i=DP::nbPermutations; i < nbRelations(); ++i) {
			m_topo[i] = relations[i];
		}
	}
};

} //namespace CGoGN

#endif
