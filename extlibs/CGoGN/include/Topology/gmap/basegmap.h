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

#ifndef __BASEGMAP_H__
#define __BASEGMAP_H__

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <list>
#include <vector>

#include "Topology/generic/genericmap.h"

namespace CGoGN
{

template < typename DART >
class tBaseGMap : public tGenericMap<DART>
{
public:
	typedef typename tGenericMap<DART>::Dart Dart;

public:
	tBaseGMap() {
	}

	tBaseGMap(const tBaseGMap<DART>& bm) : tGenericMap<DART>(bm) {
	}

	~tBaseGMap() {
		while (this->begin() != this->end()) {
			deleteDart(this->begin(),false);
		}
	}

	void deleteDart(Dart d,bool warning=true) {
		if (warning && !isFree(d)) {
			CGoGNout << "Warning: erasing a linked dart" << CGoGNendl;
		}
		tGenericMap<DART>::deleteDart(d);
	}

	//! Test if the dart is free of topological links
	/*!	@param d an iterator to the current dart
	 */
	bool isFree(Dart d) {
		for (unsigned i=0; i<DART::nbRelations(); ++i)
			if (beta(i,d) != d)
				return false;
		return true;
	}

private:
	/* A set of private basic topological operators.
	 * Translate topological links to combinatorial links.
	 */

	Dart beta_internal(Dart a, int i) const
	{
//		assert(????() || !"Only use with gmap darts");
//		ajouter qqchose dans DP pour identifier les Gcartes
		assert( ((i >=0)&&(i<DART::nbInvolutions())) || "Indices out of bounds");
		return a->getInvolution(i);
	}

	void betaSew_internal(Dart a, unsigned i, Dart d)
	{
		assert(i < DART::nbInvolutions() || !"Out of bounds");
		if (beta_internal(a,i) != d)
		{
			a->setInvolution(i,d);
		}
		else
		{
			CGoGNerr << "Warning: darts already linked with beta" << i << CGoGNendl;
		}
	}

	void betaUnsew_internal(Dart a, unsigned i)
	{
		assert(i < DART::nbInvolutions() || !"Out of bounds");
		a->unsetInvolution(i);
	}

public:
	Dart beta(unsigned int i, Dart d)
	{
		return beta_internal(d,i);
	}

	void betaSew(unsigned int i, Dart d, Dart e)
	{
		// verification GMAP ???
		betaSew_internal(d,i,e);
	}

	void betaUnsew(unsigned int i, Dart d)
	{
		// verification GMAP ???
		betaUnsew_internal(d,i);
	}

	Dart beta0(Dart d) {
		return beta(0,d);
	}
	Dart beta1(Dart d) {
		return beta(1,d);
	}
	Dart beta2(Dart d) {
		return beta(2,d);
	}
	Dart beta3(Dart d) {
		return beta(3,d);
	}
	void beta0Sew(Dart d, Dart e) {
		betaSew(0,d,e);
	}
	void beta1Sew(Dart d, Dart e) {
		betaSew(1,d,e);
	}
	void beta2Sew(Dart d, Dart e) {
		betaSew(2,d,e);
	}
	void beta3Sew(Dart d, Dart e) {
		betaSew(3,d,e);
	}

	Dart phi(int i, const Dart d)
	{
		if(i >= 0)
			return beta(i,beta(0,d)) ;
		else
		{
			unsigned j = -i ;
			return beta(0,beta(j,d)) ;
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

	Dart alpha(int i, const Dart d)
	{
		if(i >= 0)
			return beta(i,beta(DART::nbInvolutions(),d)) ;
		else
		{
			unsigned j = -i ;
			return beta(DART::nbInvolutions(),beta(i,d)) ;
		}
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

	template <int N>
	Dart beta(const Dart d) {
		if (N<10) return beta(N,d);
		return beta(N%10, beta<N/10>(d));
	}
	template<int I, int J>
	Dart beta(const Dart d) {
		return beta(J,beta(I,d));
	}
	template<int I,int J,int K>
	Dart beta(const Dart d) {
		return beta(K,beta(J,beta(I,d)));
	}
	template<int I,int J,int K, int L>
	Dart beta(const Dart d) {
		return beta(L,beta(K,beta(J,beta(I,d))));
	}

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
