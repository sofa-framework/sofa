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

#ifndef __LINEAR_SOLVING_BASIC__
#define __LINEAR_SOLVING_BASIC__

#include "NL/nl.h"
#include "Topology/generic/traversor/traversorCell.h"

namespace CGoGN
{

namespace LinearSolving
{

template <typename CoeffType>
struct Coeff
{
	unsigned int index;
	CoeffType value;
	Coeff(unsigned int i, CoeffType v) : index(i), value(v)
	{}
} ;

/*******************************************************************************
 * VARIABLES SETUP
 *******************************************************************************/

template <typename PFP, typename ATTR_TYPE>
void setupVariables(
	typename PFP::MAP& m,
	const VertexAttribute<unsigned int, typename PFP::MAP>& index,
	const CellMarker<typename PFP::MAP, VERTEX>& freeMarker,
	const VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr)
{
	foreach_cell<VERTEX>(m, [&] (Dart d)
	{
		nlSetVariable(index[d], attr[d]);
		if(!freeMarker.isMarked(d))
			nlLockVariable(index[d]);
	});
}

template <typename PFP, typename ATTR_TYPE>
void setupVariables(
	typename PFP::MAP& m,
	const VertexAttribute<unsigned int, typename PFP::MAP>& index,
	const CellMarker<typename PFP::MAP, VERTEX>& freeMarker,
	const VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr,
	unsigned int coord)
{
	foreach_cell<VERTEX>(m, [&] (Dart d)
	{
		nlSetVariable(index[d], (attr[d])[coord]);
		if(!freeMarker.isMarked(d))
			nlLockVariable(index[d]);
	});
}

/*******************************************************************************
 * MATRIX SETUP : EQUALITY
 *******************************************************************************/

template <typename PFP, typename ATTR_TYPE>
void addRowsRHS_Equality(
	typename PFP::MAP& m,
	const VertexAttribute<unsigned int, typename PFP::MAP>& index,
	const VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& weight)
{
	nlEnable(NL_NORMALIZE_ROWS) ;

	foreach_cell<VERTEX>(m, [&] (Dart d)
	{
		nlRowParameterd(NL_RIGHT_HAND_SIDE, attr[d]) ;
		nlRowParameterd(NL_ROW_SCALING, weight[d]) ;
		nlBegin(NL_ROW) ;
		nlCoefficient(index[d], 1) ;
		nlEnd(NL_ROW) ;
	});

	nlDisable(NL_NORMALIZE_ROWS) ;
}

template <typename PFP, typename ATTR_TYPE>
void addRowsRHS_Equality(
	typename PFP::MAP& m,
	const VertexAttribute<unsigned int, typename PFP::MAP>& index,
	const VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr,
	float weight)
{
	nlEnable(NL_NORMALIZE_ROWS) ;

	foreach_cell<VERTEX>(m, [&] (Dart d)
	{
		nlRowParameterd(NL_RIGHT_HAND_SIDE, attr[d]) ;
		nlRowParameterd(NL_ROW_SCALING, weight) ;
		nlBegin(NL_ROW) ;
		nlCoefficient(index[d], 1) ;
		nlEnd(NL_ROW) ;
	});

	nlDisable(NL_NORMALIZE_ROWS) ;
}

template <typename PFP, typename ATTR_TYPE>
void addRowsRHS_Equality(
	typename PFP::MAP& m,
	const VertexAttribute<unsigned int, typename PFP::MAP>& index,
	const VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& weight,
	unsigned int coord)
{
	nlEnable(NL_NORMALIZE_ROWS) ;

	foreach_cell<VERTEX>(m, [&] (Dart d)
	{
		nlRowParameterd(NL_RIGHT_HAND_SIDE, (attr[d])[coord]) ;
		nlRowParameterd(NL_ROW_SCALING, weight[d]) ;
		nlBegin(NL_ROW) ;
		nlCoefficient(index[d], 1) ;
		nlEnd(NL_ROW) ;
	});

	nlDisable(NL_NORMALIZE_ROWS) ;
}

template <typename PFP, typename ATTR_TYPE>
void addRowsRHS_Equality(
	typename PFP::MAP& m,
	const VertexAttribute<unsigned int, typename PFP::MAP>& index,
	const VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr,
	float weight,
	unsigned int coord)
{
	nlEnable(NL_NORMALIZE_ROWS) ;

	foreach_cell<VERTEX>(m, [&] (Dart d)
	{
		nlRowParameterd(NL_RIGHT_HAND_SIDE, (attr[d])[coord]) ;
		nlRowParameterd(NL_ROW_SCALING, weight) ;
		nlBegin(NL_ROW) ;
		nlCoefficient(index[d], 1) ;
		nlEnd(NL_ROW) ;
	});

	nlDisable(NL_NORMALIZE_ROWS) ;
}

/*******************************************************************************
 * MATRIX SETUP : LAPLACIAN TOPO
 *******************************************************************************/

template <typename PFP>
void addRows_Laplacian_Topo(
	typename PFP::MAP& m,
	const VertexAttribute<unsigned int, typename PFP::MAP>& index)
{
	nlEnable(NL_NORMALIZE_ROWS) ;

	foreach_cell<VERTEX>(m, [&] (Dart d)
	{
		nlRowParameterd(NL_RIGHT_HAND_SIDE, 0) ;
		nlBegin(NL_ROW);
		typename PFP::REAL aii = 0 ;
		Traversor2VE<typename PFP::MAP> t(m, d) ;
		for(Dart it = t.begin(); it != t.end(); it = t.next())
		{
			typename PFP::REAL aij = 1 ;
			aii += aij ;
			nlCoefficient(index[m.phi1(it)], aij) ;
		}
		nlCoefficient(index[d], -aii) ;
		nlEnd(NL_ROW) ;
	});

	nlDisable(NL_NORMALIZE_ROWS) ;
}

template <typename PFP, typename ATTR_TYPE>
void addRowsRHS_Laplacian_Topo(
	typename PFP::MAP& m,
	const VertexAttribute<unsigned int, typename PFP::MAP>& index,
	const VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr)
{
	nlEnable(NL_NORMALIZE_ROWS) ;

	foreach_cell<VERTEX>(m, [&] (Dart d)
	{
		std::vector<Coeff<typename PFP::REAL> > coeffs ;
		coeffs.reserve(12) ;

		typename PFP::REAL norm2 = 0 ;
		typename PFP::REAL aii = 0 ;
		Traversor2VE<typename PFP::MAP> t(m, d) ;
		for(Dart it = t.begin(); it != t.end(); it = t.next())
		{
			typename PFP::REAL aij = 1 ;
			aii += aij ;
			coeffs.push_back(Coeff<typename PFP::REAL>(index[m.phi1(it)], aij)) ;
			norm2 += aij * aij ;
		}
		coeffs.push_back(Coeff<typename PFP::REAL>(index[d], -aii)) ;
		norm2 += aii * aii ;

		nlRowParameterd(NL_RIGHT_HAND_SIDE, attr[d] * sqrt(norm2)) ;
		nlBegin(NL_ROW);
		for(unsigned int i = 0; i < coeffs.size(); ++i)
			nlCoefficient(coeffs[i].index, coeffs[i].value) ;
		nlEnd(NL_ROW) ;
	});

	nlDisable(NL_NORMALIZE_ROWS) ;
}

template <typename PFP, typename ATTR_TYPE>
void addRowsRHS_Laplacian_Topo(
	typename PFP::MAP& m,
	const VertexAttribute<unsigned int, typename PFP::MAP>& index,
	const VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr,
	unsigned int coord)
{
	nlEnable(NL_NORMALIZE_ROWS) ;

	foreach_cell<VERTEX>(m, [&] (Dart d)
	{
		std::vector<Coeff<typename PFP::REAL> > coeffs ;
		coeffs.reserve(12) ;

		typename PFP::REAL norm2 = 0 ;
		typename PFP::REAL aii = 0 ;
		Traversor2VE<typename PFP::MAP> t(m, d) ;
		for(Dart it = t.begin(); it != t.end(); it = t.next())
		{
			typename PFP::REAL aij = 1 ;
			aii += aij ;
			coeffs.push_back(Coeff<typename PFP::REAL>(index[m.phi1(it)], aij)) ;
			norm2 += aij * aij ;
		}
		coeffs.push_back(Coeff<typename PFP::REAL>(index[d], -aii)) ;
		norm2 += aii * aii ;

		nlRowParameterd(NL_RIGHT_HAND_SIDE, (attr[d])[coord] * sqrt(norm2)) ;
		nlBegin(NL_ROW);
		for(unsigned int i = 0; i < coeffs.size(); ++i)
			nlCoefficient(coeffs[i].index, coeffs[i].value) ;
		nlEnd(NL_ROW) ;
	});

	nlDisable(NL_NORMALIZE_ROWS) ;
}

/*******************************************************************************
 * MATRIX SETUP : LAPLACIAN COTAN
 *******************************************************************************/

template <typename PFP>
void addRows_Laplacian_Cotan(
	typename PFP::MAP& m,
	const VertexAttribute<unsigned int, typename PFP::MAP>& index,
	const EdgeAttribute<typename PFP::REAL, typename PFP::MAP>& edgeWeight,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& vertexArea)
{
	nlEnable(NL_NORMALIZE_ROWS) ;

	foreach_cell<VERTEX>(m, [&] (Dart d)
	{
		nlRowParameterd(NL_RIGHT_HAND_SIDE, 0) ;
		nlBegin(NL_ROW);
		typename PFP::REAL vArea = vertexArea[d] ;
		typename PFP::REAL aii = 0 ;
		Traversor2VE<typename PFP::MAP> t(m, d) ;
		for(Dart it = t.begin(); it != t.end(); it = t.next())
		{
			typename PFP::REAL aij = edgeWeight[it] / vArea ;
			aii += aij ;
			nlCoefficient(index[m.phi1(it)], aij) ;
		}
		nlCoefficient(index[d], -aii) ;
		nlEnd(NL_ROW) ;
	});

	nlDisable(NL_NORMALIZE_ROWS) ;
}

template <typename PFP, typename ATTR_TYPE>
void addRowsRHS_Laplacian_Cotan(
	typename PFP::MAP& m,
	const VertexAttribute<unsigned int, typename PFP::MAP>& index,
	const EdgeAttribute<typename PFP::REAL, typename PFP::MAP>& edgeWeight,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& vertexArea,
	const VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr)
{
	nlEnable(NL_NORMALIZE_ROWS) ;

	foreach_cell<VERTEX>(m, [&] (Dart d)
	{
		std::vector<Coeff<typename PFP::REAL> > coeffs ;
		coeffs.reserve(12) ;

		typename PFP::REAL vArea = vertexArea[d] ;
		typename PFP::REAL norm2 = 0 ;
		typename PFP::REAL aii = 0 ;
		Traversor2VE<typename PFP::MAP> t(m, d) ;
		for(Dart it = t.begin(); it != t.end(); it = t.next())
		{
			typename PFP::REAL aij = edgeWeight[it] / vArea ;
			aii += aij ;
			coeffs.push_back(Coeff<typename PFP::REAL>(index[m.phi1(it)], aij)) ;
			norm2 += aij * aij ;
		}
		coeffs.push_back(Coeff<typename PFP::REAL>(index[d], -aii)) ;
		norm2 += aii * aii ;

		nlRowParameterd(NL_RIGHT_HAND_SIDE, attr[d] * sqrt(norm2)) ;
		nlBegin(NL_ROW);
		for(unsigned int i = 0; i < coeffs.size(); ++i)
			nlCoefficient(coeffs[i].index, coeffs[i].value) ;
		nlEnd(NL_ROW) ;
	});

	nlDisable(NL_NORMALIZE_ROWS) ;
}

template <typename PFP, typename ATTR_TYPE>
void addRowsRHS_Laplacian_Cotan(
	typename PFP::MAP& m,
	const VertexAttribute<unsigned int, typename PFP::MAP>& index,
	const EdgeAttribute<typename PFP::REAL, typename PFP::MAP>& edgeWeight,
	const VertexAttribute<typename PFP::REAL, typename PFP::MAP>& vertexArea,
	const VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr,
	unsigned int coord)
{
	nlEnable(NL_NORMALIZE_ROWS) ;

	foreach_cell<VERTEX>(m, [&] (Dart d)
	{
		std::vector<Coeff<typename PFP::REAL> > coeffs ;
		coeffs.reserve(12) ;

		typename PFP::REAL vArea = vertexArea[d] ;
		typename PFP::REAL norm2 = 0 ;
		typename PFP::REAL aii = 0 ;
		Traversor2VE<typename PFP::MAP> t(m, d) ;
		for(Dart it = t.begin(); it != t.end(); it = t.next())
		{
			typename PFP::REAL aij = edgeWeight[it] / vArea ;
			aii += aij ;
			coeffs.push_back(Coeff<typename PFP::REAL>(index[m.phi1(it)], aij)) ;
			norm2 += aij * aij ;
		}
		coeffs.push_back(Coeff<typename PFP::REAL>(index[d], -aii)) ;
		norm2 += aii * aii ;

		nlRowParameterd(NL_RIGHT_HAND_SIDE, (attr[d])[coord] * sqrt(norm2)) ;
		nlBegin(NL_ROW);
		for(unsigned int i = 0; i < coeffs.size(); ++i)
			nlCoefficient(coeffs[i].index, coeffs[i].value) ;
		nlEnd(NL_ROW) ;
	});

	nlDisable(NL_NORMALIZE_ROWS) ;
}

/*******************************************************************************
 * GET RESULTS
 *******************************************************************************/

template <typename PFP, typename ATTR_TYPE>
void getResult(
	typename PFP::MAP& m,
	const VertexAttribute<unsigned int, typename PFP::MAP>& index,
	VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr)
{
	foreach_cell<VERTEX>(m, [&] (Dart d)
	{
		attr[d] = nlGetVariable(index[d]) ;
	});
}

template <typename PFP, typename ATTR_TYPE>
void getResult(
	typename PFP::MAP& m,
	const VertexAttribute<unsigned int, typename PFP::MAP>& index,
	VertexAttribute<ATTR_TYPE, typename PFP::MAP>& attr,
	unsigned int coord)
{
	foreach_cell<VERTEX>(m, [&] (Dart d)
	{
		(attr[d])[coord] = nlGetVariable(index[d]) ;
	});
}

} // namespace LinearSolving

} // namespace CGoGN

#endif
