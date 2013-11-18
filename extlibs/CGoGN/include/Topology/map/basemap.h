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

#ifndef __BASEMAP_H__
#define __BASEMAP_H__

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <list>
#include <vector>

#include "Topology/generic/genericmap.h"

namespace CGoGN
{

class BaseMap : public GenericMap
{
public:
	typedef GenericMap::Dart Dart;

	BaseMap() : GenericMap()
	{
		dual_linking = true ;
	}
	
	BaseMap(const BaseMap& bm) : GenericMap(bm)
	{
		dual_linking = bm.isDual() ;
	}

	bool isDual() const { return dual_linking ; }
	bool isPrimal() const { return !dual_linking ; }

	void deleteDart(Dart d, bool warning=true)
	{
		if (warning && !isFree(d)) {
			CGoGNout << "Warning: erasing a linked dart" << CGoGNendl;
		}
		GenericMap::deleteDart(d);
	}

	//! Test if the dart is free of topological links
	/*!	@param d an iterator to the current dart
	 */
	bool isFree(Dart d)
	{
		if (isDual())
		{
			for (unsigned int i = 1; i <= nbRelations(); ++i)
				if (phi(i,d) != d)
					return false;
		}
		else
		{
			for (unsigned int i = 0; i < nbRelations(); ++i)
				if (alpha(i,d) != d)
					return false;
		}
		return true;
	}

private:
	bool dual_linking ;

	/*
	 * These private operators return a Dart obtained from the basic operators depending on
	 * the dual or primal encoding of topological relations. The relations between high-level
	 * and basic operators is set during the compilation.
	 *
	 * The inverse are obtained with negative indexes:
	 * - alpha(-i)	= inverse of alpha(i)	with i in [1,N[
	 * - alpha(-N)	= inverse of alpha(0)	(as -0 == 0)
	 * - phi(-i)	= inverse of phi(i)		with i in [1,N]
	 * - phi(0) is not defined
	 *
	 * These operators respect the following relations
	 * (if N = nbRelation and * is the composition of function):
	 * - alpha(0)	= phi(N)
	 * - alpha(i)	= phi(i)*phi(N)
	 * - alpha(-N) 	= phi(-N)
	 * - alpha(-i)	= phi(-N)*phi(-i)
	 *
	 * And conversely :
	 * - phi(N)		= alpha(0)
	 * - phi(i)		= alpha(i)*alpha(0)
	 * - phi(-N)	= alpha(-N)
	 * - phi(-i)	= alpha(-N)*alpha(-i)
	 */

	//! Get primal topological relation or its inverse
	/*!	@param i in [0, nbInvolutions[				: get an involution
	 *	@param i in [nbInvolutions, nbRelations[	: get a permutation
	 *	@param i in ]-nbInvolutions, 0[				: get the inverse of an involution
	 *	@param i in ]-nbRelations, -nbInvolutions[	: get the inverse of a permutation
	 *	@param i is -nbRelations					: get the inverse of alpha(0)
	 */
	Dart alpha_internal(Dart a, int i) const {
		assert(isPrimal() || !"Only use with primal darts");
		if (i >= 0) {
			unsigned int j = (unsigned int) i;
			assert(j < nbRelations() || !"Out of bounds");
			if (j < m_nbInvolutions)
				return getInvolution(a,j);
			else
				return getPermutation(a,j-m_nbInvolutions);
		}
		else {
			unsigned int j = (unsigned int) -i;
			if (j == nbRelations()) j = 0;
			assert(j < nbRelations() || !"Out of bounds");
			if (j < m_nbInvolutions)
				return getInvolution(a,j);
			else
				return getPermutationInverse(a,j-m_nbInvolutions);
		}
	}

	//! Link dart a to dart d with a primal topological relation
	/*!	@param i in [0, nbInvolutions[				: set the involutions
	 *	@param i in [nbInvolutions, nbRelations[	: set the permutations
	 *  @param a the dart to link
	 *  @param d the dart to which dart a is linked
	 * If the topological link already exists, then do nothing
	 */
	void alphaSew_internal(Dart a, unsigned int i, Dart d) {
		assert(isPrimal() || !"Only use with primal darts");
		assert(i < nbRelations() || !"Out of bounds");
		if (alpha_internal(a,i) != d) {
			if (i < m_nbInvolutions)
				setInvolution(a,d,i);
			else
				setPermutation(a,d,i-m_nbInvolutions);
		}
		else {
			CGoGNerr << "Warning: darts already linked with alpha" << i << CGoGNendl;
		}
	}

	//! Unlink dart a from its successor in a primal topological relation
	/*!	@param i in [0, nbInvolutions[				: set the involutions
	 *	@param i in [nbInvolutions, nbRelations[	: set the permutations
	 */
	void alphaUnsew_internal(Dart a, unsigned int i) {
		assert(isPrimal() || !"Only use with primal darts");
		assert(i < nbRelations() || !"Out of bounds");
		if (i < m_nbInvolutions)
			unsetInvolution(a,i);
		else
			unsetPermutation(a,i-m_nbInvolutions);
	}

	//! Get dual topological relation
	/*!	@param i in [1, nbPermutations]				: get a permutation
	 *	@param i in ]nbPermutations, nbRelations]	: get an involution
	 *	@param i in [-nbPermutations, -1[			: get the inverse of a permutation
	 *	@param i in [-nbRelations, -nbPermutations[	: get the inverse of an involution
	 *  The value 0 (zero) is not allowed for i
	 */
	Dart phi_internal(Dart a, int i) const {
		assert(isDual() || !"Only use with dual darts");
		if (i >= 0) {
			unsigned int j = (unsigned int) i;
			assert(j <= nbRelations() || !"Out of bounds");
			if (j <= m_nbPermutations)
				return getPermutation(a,j-1);
			else
				return getInvolution(a,j-1-m_nbPermutations);
		}
		else {
			unsigned int j = (unsigned int) -i;
			assert(j <= nbRelations() || !"Out of bounds");
			if (j <= m_nbPermutations)
				return getPermutationInverse(a,j-1);
			else
				return getInvolution(a,j-1-m_nbPermutations);
		}
	}

	//! Link dart a to dart d with a dual topological relation
	/*!	@param i in [0, nbPermutations[				: get a permutation
	 *	@param i in [nbPermutations, nbRelations[	: get an involution
	 *  @param d the dart to which the current is linked
	 */
	void phiSew_internal(Dart a, unsigned int i, Dart d) {
		assert(isDual() || !"Only use with dual darts");
		assert(i <= nbRelations() || !"Out of bounds");
		if (phi_internal(a,i) != d) {
			if (i <= m_nbPermutations)
				setPermutation(a,d,i-1);
			else
				setInvolution(a,d,i-1-m_nbPermutations);
		}
		else {
			CGoGNerr << "Warning: darts already linked with phi" << i << CGoGNendl;
		}
	}

	//! Unlink the current dart from its successor in a dual topological relation
	/*!	@param i in [0, nbPermutations[				: get a permutation
	 *	@param i in [nbPermutations, nbRelations[	: get an involution
	 * - Before:	c->d and d->e		(c is the current dart = an iterator to this)
	 * - After:		c->e and d->d		(e=c if the relation is an involution)
	 */
	void phiUnsew_internal(Dart a, unsigned int i) {
		assert(isDual() || !"Only use with dual darts");
		assert(i <= nbRelations() || !"Out of bounds");
		if (i <= m_nbPermutations)
			unsetPermutation(a,i-1);
		else
			unsetInvolution(a,i-1-m_nbPermutations);
	}

public:
	//! Get primal topological relation
	/*!	@param i in [0, nbRelations[	: get the primal relation
	 *	@param i in [-nbRelations, 0[	: get the inverse of a primal relation
	 *  @param d the dart to look at
	 *  The inverse of alpha(0) is accessed with alpha(-nbRelations)
	 *  (this is only usefull if alpha(O) is a permutation, ie if there is no involution)
	 */
	Dart alpha(int i, const Dart d) {
		if (isPrimal())
			return alpha_internal(d,i);

		// Dual DART: alpha is defined from phi
		if (i >= 0) {
			if (i == 0) {
				return phi_internal(d,nbRelations());
			}
			else { // i > 0
				Dart dPhiN = phi_internal(d,nbRelations());
				if (dPhiN != d)
					return phi_internal(dPhiN,i);
				// d is a boundary dart => search its neighbour
				Dart current = d;
				Dart dPrev = phi_internal(current,-i);
				dPhiN = phi_internal(dPrev,-int(nbRelations()));
				while (dPhiN != dPrev) {
					current = dPhiN;
					dPrev = phi_internal(current,-i);
					dPhiN = phi_internal(dPrev,-int(nbRelations()));
				} 
				return current;
			}
		}
		else { // i < 0
			if ((unsigned int)-i == nbRelations()) {
				return phi_internal(d,-int(nbRelations()));
			}
			else { // i < 0
				Dart dPrev = phi_internal(d,i);
				Dart dPhiN = phi_internal(dPrev,-int(nbRelations()));
				if (dPrev != dPhiN)
					return dPhiN;
				// d is a boundary dart => search its neighbour
				Dart current = d;
				dPhiN = phi_internal(current,nbRelations());
				while (current != dPhiN) {
					current = phi_internal(dPhiN,-i);
					dPhiN = phi_internal(current,nbRelations());
				};
				return current;
			}
		}
	}

	//! Link the current dart to dart d through relation alpha(i)
	/*!	@param i in [0, nbRelations[
	 *  @param d the dart to link
	 *  @param e the dart to which d is linked
	 */
	void alphaSew(int i, Dart d, Dart e) {
		if (isPrimal())
			alphaSew_internal(d,i,e);
		else {
			assert(!"Not implemented");
		}
	}

	//! Unlink the current dart from its successor through relation alpha(i)
	void alphaUnsew(int i, Dart d) {
		if (isPrimal())
			alphaUnsew_internal(d,i);
		else {
			assert(!"Not implemented");
		}
	}
	
	//! Get dual topological relation
	/*!	@param i in [1, nbRelations]	: get the dual relation
	 *	@param i in [-nbRelations, -1]	: get the inverse of a dual relation
	 *  @param d the dart to look at
	 * The value 0 (zero) is not allowed for i
	 */
	Dart phi(int i, const Dart d) {
		if (isDual())
			return phi_internal(d,i);

		// Primal DART: phi is defined from alpha
		if (i >= 0) {
			if ((unsigned int)i == nbRelations())
				return alpha(0,d);
			else
				return alpha(i,alpha(0,d));
		}
		else { // i < 0
			if ((unsigned int)-i == nbRelations())
				return alpha(-int(nbRelations()),d);
			else
				return alpha(-int(nbRelations()),alpha(i,d));
		}
	}

	//! Link dart d to dart e through relation phi(i)
	/*!	@param i in [1, nbRelations]
	 *  @param d the first dart to link
	 *  @param e the second dart to link
	 */
	void phiSew(int i,Dart d, Dart e) {
		if (isDual())
			phiSew_internal(d,i,e);
		else {
			assert(!"Not implemented");
		}
	}

	//! Unlink dart d from its successor through relation phi(i)
	/*!	@param i in [1, nbRelations]
	 *  @param d the dart to unsew
	 */
	void phiUnsew(int i,Dart d) {
		if (isDual())
			phiUnsew_internal(d,i);
		else {
			assert(!"Not implemented");
		}
	}

	Dart phi1( const Dart d) {
		return phi( 1,d);
	}
	Dart phi_1(const Dart d) {
		return phi_1(d);
	}
	Dart phi2( const Dart d) {
		return phi( 2,d);
	}
	Dart phi_2(const Dart d) {
		return phi(-2,d);
	}
	Dart phi3( const Dart d) {
		return phi( 3,d);
	}
	Dart phi_3(const Dart d) {
		return phi(-3,d);
	}
	Dart phi4( const Dart d) {
		return phi( 4,d);
	}
	Dart phi_4(const Dart d) {
		return phi(-4,d);
	}
	void phi1sew(Dart d, Dart e) {
		phi1sew(d,e);
	}
	void phi2sew(Dart d, Dart e) {
		phi2sew(d,e);
	}
	void phi3Sew(Dart d, Dart e) {
		phiSew(3,d,e);
	}
	void phi4Sew(Dart d, Dart e) {
		phiSew(4,d,e);
	}

	Dart alpha1( const Dart d) {
		return alpha( 1,d);
	}
	Dart alpha_1(const Dart d) {
		return alpha(-1,d);
	}
	Dart alpha2( const Dart d) {
		return alpha( 2,d);
	}
	Dart alpha_2(const Dart d) {
		return alpha(-2,d);
	}
	Dart alpha3( const Dart d) {
		return alpha( 3,d);
	}
	Dart alpha_3(const Dart d) {
		return alpha(-3,d);
	}
	Dart alpha4( const Dart d) {
		return alpha( 4,d);
	}
	Dart alpha_4(const Dart d) {
		return alpha(-4,d);
	}
	void alpha1Sew(Dart d, Dart e) {
		alphaSew(1,d,e);
	}
	void alpha2Sew(Dart d, Dart e) {
		alphaSew(2,d,e);
	}
	void alpha3Sew(Dart d, Dart e) {
		alphaSew(3,d,e);
	}
	void alpha4Sew(Dart d, Dart e) {
		alphaSew(4,d,e);
	}

	/**
	* Apply the phi in same order
	* Ex: phi<21>(d) <-> phi1(phi2(d))
	* But can not work with inverse phi !!
	*/
	template <int N>
	Dart phi(const Dart d) {
		assert( (N >0 ) || !"negative parameters not allowed in template multi-phi");
		if (N<10) return phi(N,d);
		return phi(N%10, phi<N/10>(d));
	}
	template<int I, int J>
	Dart phi(const Dart d) {
		return phi(J,phi(I,d));
	}
	template<int I, int J, int K>
	Dart phi(const Dart d) {
		return phi(K,phi(J,phi(I,d)));
	}
	template<int I, int J, int K , int L>
	Dart phi(const Dart d) {
		return phi(L,phi(K,phi(J,phi(I,d))));
	}
};

} //namespace CGoGN

#endif
