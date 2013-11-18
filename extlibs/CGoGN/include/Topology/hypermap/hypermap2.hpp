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

namespace CGoGN
{

template <typename DART>
typename tHyperMap<DART>::Dart tHyperMap<DART>::newHyperDart()
{
	push_front(DART());	// push front because is is more easy to return the iterator
	Dart d= this->begin();
	d->initlab(NIL,m_nextLabel);	// initialize the dart with NIL (end iterator)
	m_nextLabel++;
	m_nbDarts++;
	return d;
}

template <typename DART>
typename tHyperMap<DART>::Dart tHyperMap<DART>::alpha(int i, Dart d)
{
	return d->get_permutation(i);
}

template <typename DART>
typename tHyperMap<DART>::Dart tHyperMap<DART>::alpha_(int i, Dart d)
{
	return d->get_permutation_inv(i);
}

template <typename DART>
void tHyperMap<DART>::sewAlpha(int i, Dart d, Dart e)
{
	d->set_permutation(i,e);
	e->set_permutation_inv(i,d);
}

template <typename DART>
void tHyperMap<DART>::unsewAlpha(int i, Dart d)
{
	Dart e = alpha(i,d);
	Dart f = alpha_(i,d);
	e->set_permutation_inv(i,NIL);
	f->set_permutation(i,NIL);
	d->set_permutation(i,NIL);
	d->set_permutation_inv(i,NIL);
}

template <typename DART>
typename tHyperMap<DART>::Dart tHyperMap<DART>::createPseudoEdge()
{
	Dart d = newHyperDart();
	Dart e = newHyperDart();
	sewAlpha(0,d,e);
	return  d;
}

template <typename DART>
typename tHyperMap<DART>::Dart tHyperMap<DART>::createFace(int nbEdges)
{
	int nbd = nbEdges/2;

	// create the first "edge"
	Dart e0 = createPseudoEdge();
	Dart e1 = e0;
	
	for (int i=1; i<nbEdges; i++)
	{
		Dart e2 = createPseudoEdge; // create the next edge
		sewalpha1( alpha(0,e1), e2);
		e1 = e2;	// change the preceeding
	}
	sewalpha1(alpha(0,e1), e0);

	return e0;
}

template <typename DART>
void tHyperMap<DART>::edgeFusion(Dart d, Dart e) 
{
	Dart ee = alpha1(d);
	sewalpha1(d,ee);

	Dart ee = alpha_(0,e);
	sewAlpha(0,ee,d);

	deleteDart(e);
}

template <typename DART>
template <typename MAP>
bool tHyperMap<DART>::foreach_dart_of_face(Dart d, FunctorType<MAP>& f)
{
	Dart e = d;
	do
	{
		if (f(e)) return true;
		e = alpha_(0,e);
		e = alpha_(1,e); // orbit face
	} while (e!=d);
	return false;
}

} // namespace CGoGN

