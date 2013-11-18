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

template <typename MF, typename MM>
void markOrbitGen(int dim, typename MM::Dart d, Marker m, unsigned int th)
{
	FunctorMark<MM> f(m);
	MF::foreach_dart_of_orbit(dim, d, f, th);
}

template <typename MF, typename MM>
void unmarkOrbitGen(int dim, typename MM::Dart d, Marker m, unsigned int th)
{
	FunctorUnmark<MM> f(m);
	MF::foreach_dart_of_orbit(dim, d, f, th);
}

template <typename MF, typename MM>
	void foreach_orbitGen_sel(int dim, FunctorType<typename MM>& f, MM* ptr, FunctorType<typename MM>& good, unsigned int th )
{
	// lock a marker
	DartMarker markerCell(*ptr,th);

	// scan all dart of the map
	for(typename MM::Dart d = ptr->begin(); d != ptr->end(); ptr->next(d))
	{
		if (good(d))
		{
			if (!markerCell.isMarked(d))  // if not marked
			{
				if ((f)(d))			// call the functor and
				{
					d=ptr->end();
					--d;
				}
				else
					markOrbitGen<MF,MM>(dim, d, markerCell, ptr, th);  // mark all dart of the vertex
			}
		}
	}
}


template <typename MF, typename MM>
void foreach_orbitGen(int dim, FunctorType<typename MM>& fonct, MM* ptr, unsigned int th)
{
	// lock a marker
//	Marker markerCell = ptr->getNewMarker();
//
//	// scan all dart of the map
//	for(typename MM::Dart d = ptr->begin(); d != ptr->end(); ptr->next(d))
//	{
//		if (!ptr->isMarkedDart(d,markerCell))  // if not marked
//		{
//			if ((fonct)(d))			// call the functor and
//			{
//				d=ptr->end();
//				--d;
//			}
//			else
//				markOrbitGen<MF,MM>(dim,d,markerCell,ptr);  // mark all dart of the vertex
//		}
//	}
//	ptr->unmarkAll(markerCell);
//	ptr->releaseMarker(markerCell);
}

template <typename MF, typename MM>
void setOrbitEmbeddingGen(int dim, typename MM::Dart d, int index, Embedding* em, MM* ptr)
{
//	if (em!=NULL)
//	{
//		FunctorSetEmb<typename MM::Dart> fse(index,em);
//		ptr->foreach_dart_of_orbit(dim, d, fse);
//	}
//	else
//	{
//		FunctorUnsetEmb<typename MM::Dart> fse(index);
//		ptr->foreach_dart_of_orbit(dim, d, fse);
//	}
}

} //namespace CGoGN
