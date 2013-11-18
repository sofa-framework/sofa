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

#ifndef __LINEAR_SOLVING_MATRIX_SETUP__
#define __LINEAR_SOLVING_MATRIX_SETUP__

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
 * EQUALITY MATRIX : right-hand-side SCALAR
 *******************************************************************************/

template<typename PFP, typename ATTR_TYPE>
class FunctorEquality_PerVertexWeight_Scalar : public FunctorType
{
protected:
	const VertexAttribute<unsigned int>& indexTable ;
	const VertexAttribute<ATTR_TYPE>& attrTable ;
	const VertexAttribute<typename PFP::REAL>& weightTable ;

public:
	FunctorEquality_PerVertexWeight_Scalar(
		const VertexAttribute<unsigned int>& index,
		const VertexAttribute<ATTR_TYPE>& attr,
		const VertexAttribute<typename PFP::REAL>& weight
	) :	indexTable(index), attrTable(attr), weightTable(weight)
	{}

	bool operator()(Dart d)
	{
		nlRowParameterd(NL_RIGHT_HAND_SIDE, attrTable[d]) ;
		nlRowParameterd(NL_ROW_SCALING, weightTable[d]) ;
		nlBegin(NL_ROW) ;
		nlCoefficient(indexTable[d], 1) ;
		nlEnd(NL_ROW) ;
		return false ;
	}
} ;

template<typename PFP, typename ATTR_TYPE>
class FunctorEquality_UniformWeight_Scalar : public FunctorType
{
protected:
	const VertexAttribute<unsigned int>& indexTable ;
	const VertexAttribute<ATTR_TYPE>& attrTable ;
	float weight ;

public:
	FunctorEquality_UniformWeight_Scalar(
		const VertexAttribute<unsigned int>& index,
		const VertexAttribute<ATTR_TYPE>& attr,
		float w
	) :	indexTable(index), attrTable(attr), weight(w)
	{}

	bool operator()(Dart d)
	{
		nlRowParameterd(NL_RIGHT_HAND_SIDE, attrTable[d]) ;
		nlRowParameterd(NL_ROW_SCALING, weight) ;
		nlBegin(NL_ROW) ;
		nlCoefficient(indexTable[d], 1) ;
		nlEnd(NL_ROW) ;
		return false ;
	}
} ;

/*******************************************************************************
 * EQUALITY MATRIX : right-hand-side VECTOR + coordinate
 *******************************************************************************/

template<typename PFP, typename ATTR_TYPE>
class FunctorEquality_PerVertexWeight_Vector : public FunctorType
{
protected:
	const VertexAttribute<unsigned int>& indexTable ;
	const VertexAttribute<ATTR_TYPE>& attrTable ;
	const VertexAttribute<typename PFP::REAL>& weightTable ;
	unsigned int coord ;

public:
	FunctorEquality_PerVertexWeight_Vector(
		const VertexAttribute<unsigned int>& index,
		const VertexAttribute<ATTR_TYPE>& attr,
		const VertexAttribute<typename PFP::REAL>& weight,
		unsigned int c
	) :	indexTable(index), attrTable(attr), weightTable(weight), coord(c)
	{}

	bool operator()(Dart d)
	{
		nlRowParameterd(NL_RIGHT_HAND_SIDE, (attrTable[d])[coord]) ;
		nlRowParameterd(NL_ROW_SCALING, weightTable[d]) ;
		nlBegin(NL_ROW) ;
		nlCoefficient(indexTable[d], 1) ;
		nlEnd(NL_ROW) ;
		return false ;
	}
} ;

template<typename PFP, typename ATTR_TYPE>
class FunctorEquality_UniformWeight_Vector : public FunctorType
{
protected:
	const VertexAttribute<unsigned int>& indexTable ;
	const VertexAttribute<ATTR_TYPE>& attrTable ;
	float weight ;
	unsigned int coord ;

public:
	FunctorEquality_UniformWeight_Vector(
		const VertexAttribute<unsigned int>& index,
		const VertexAttribute<ATTR_TYPE>& attr,
		float w,
		unsigned int c
	) :	indexTable(index), attrTable(attr), weight(w), coord(c)
	{}

	bool operator()(Dart d)
	{
		nlRowParameterd(NL_RIGHT_HAND_SIDE, (attrTable[d])[coord]) ;
		nlRowParameterd(NL_ROW_SCALING, weight) ;
		nlBegin(NL_ROW) ;
		nlCoefficient(indexTable[d], 1) ;
		nlEnd(NL_ROW) ;
		return false ;
	}
} ;

/*******************************************************************************
 * LAPLACIAN TOPO MATRIX : right-hand-side 0
 *******************************************************************************/

template<typename PFP>
class FunctorLaplacianTopo : public FunctorMap<typename PFP::MAP>
{
protected:
	const VertexAttribute<unsigned int>& indexTable ;

public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::REAL REAL ;

	FunctorLaplacianTopo(
		MAP& m,
		const VertexAttribute<unsigned int>& index
	) :	FunctorMap<MAP>(m), indexTable(index)
	{}

	bool operator()(Dart d)
	{
		nlRowParameterd(NL_RIGHT_HAND_SIDE, 0) ;
		nlBegin(NL_ROW);
		REAL aii = 0 ;
		Traversor2VE<typename PFP::MAP> t(this->m_map, d) ;
		for(Dart it = t.begin(); it != t.end(); it = t.next())
		{
			REAL aij = 1 ;
			aii += aij ;
			nlCoefficient(indexTable[this->m_map.phi1(it)], aij) ;
		}
		nlCoefficient(indexTable[d], -aii) ;
		nlEnd(NL_ROW) ;
		return false ;
	}
} ;

/*******************************************************************************
 * LAPLACIAN TOPO MATRIX : right-hand-side SCALAR
 *******************************************************************************/

template<typename PFP, typename ATTR_TYPE>
class FunctorLaplacianTopoRHS_Scalar : public FunctorMap<typename PFP::MAP>
{
protected:
	const VertexAttribute<unsigned int>& indexTable ;
	const VertexAttribute<ATTR_TYPE>& attrTable ;

public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::REAL REAL ;

	FunctorLaplacianTopoRHS_Scalar(
		MAP& m,
		const VertexAttribute<unsigned int>& index,
		const VertexAttribute<ATTR_TYPE>& attr
	) :	FunctorMap<MAP>(m), indexTable(index), attrTable(attr)
	{}

	bool operator()(Dart d)
	{
		std::vector<Coeff<REAL> > coeffs ;
		coeffs.reserve(12) ;

		REAL norm2 = 0 ;
		REAL aii = 0 ;
		Traversor2VE<typename PFP::MAP> t(this->m_map, d) ;
		for(Dart it = t.begin(); it != t.end(); it = t.next())
		{
			REAL aij = 1 ;
			aii += aij ;
			coeffs.push_back(Coeff<REAL>(indexTable[this->m_map.phi1(it)], aij)) ;
			norm2 += aij * aij ;
		}
		coeffs.push_back(Coeff<REAL>(indexTable[d], -aii)) ;
		norm2 += aii * aii ;

		nlRowParameterd(NL_RIGHT_HAND_SIDE, attrTable[d] * sqrt(norm2)) ;
		nlBegin(NL_ROW);
		for(unsigned int i = 0; i < coeffs.size(); ++i)
			nlCoefficient(coeffs[i].index, coeffs[i].value) ;
		nlEnd(NL_ROW) ;

		return false ;
	}
} ;

/*******************************************************************************
 * LAPLACIAN TOPO MATRIX : right-hand-side VECTOR + coordinate
 *******************************************************************************/

template<typename PFP, typename ATTR_TYPE>
class FunctorLaplacianTopoRHS_Vector : public FunctorMap<typename PFP::MAP>
{
protected:
	const VertexAttribute<unsigned int>& indexTable ;
	const VertexAttribute<ATTR_TYPE>& attrTable ;
	unsigned int coord ;

public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::REAL REAL ;

	FunctorLaplacianTopoRHS_Vector(
		MAP& m,
		const VertexAttribute<unsigned int>& index,
		const VertexAttribute<ATTR_TYPE>& attr,
		unsigned int c
	) :	FunctorMap<MAP>(m), indexTable(index), attrTable(attr), coord(c)
	{}

	bool operator()(Dart d)
	{
		std::vector<Coeff<REAL> > coeffs ;
		coeffs.reserve(12) ;

		REAL norm2 = 0 ;
		REAL aii = 0 ;
		Traversor2VE<typename PFP::MAP> t(this->m_map, d) ;
		for(Dart it = t.begin(); it != t.end(); it = t.next())
		{
			REAL aij = 1 ;
			aii += aij ;
			coeffs.push_back(Coeff<REAL>(indexTable[this->m_map.phi1(it)], aij)) ;
			norm2 += aij * aij ;
		}
		coeffs.push_back(Coeff<REAL>(indexTable[d], -aii)) ;
		norm2 += aii * aii ;

		nlRowParameterd(NL_RIGHT_HAND_SIDE, (attrTable[d])[coord] * sqrt(norm2)) ;
		nlBegin(NL_ROW);
		for(unsigned int i = 0; i < coeffs.size(); ++i)
			nlCoefficient(coeffs[i].index, coeffs[i].value) ;
		nlEnd(NL_ROW) ;

		return false ;
	}
} ;

/*******************************************************************************
 * LAPLACIAN COTAN MATRIX : right-hand-side 0
 *******************************************************************************/

template<typename PFP>
class FunctorLaplacianCotan : public FunctorMap<typename PFP::MAP>
{
protected:
	const VertexAttribute<unsigned int>& indexTable ;
	const EdgeAttribute<typename PFP::REAL>& edgeWeight ;
	const VertexAttribute<typename PFP::REAL>& vertexArea ;

public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::REAL REAL ;

	FunctorLaplacianCotan(
		MAP& m,
		const VertexAttribute<unsigned int>& index,
		const EdgeAttribute<typename PFP::REAL>& eWeight,
		const VertexAttribute<typename PFP::REAL>& vArea
	) :	FunctorMap<MAP>(m), indexTable(index), edgeWeight(eWeight), vertexArea(vArea)
	{}

	bool operator()(Dart d)
	{
		nlRowParameterd(NL_RIGHT_HAND_SIDE, 0) ;
		nlBegin(NL_ROW);
		REAL vArea = vertexArea[d] ;
		REAL aii = 0 ;
		Traversor2VE<typename PFP::MAP> t(this->m_map, d) ;
		for(Dart it = t.begin(); it != t.end(); it = t.next())
		{
			REAL aij = edgeWeight[it] / vArea ;
			aii += aij ;
			nlCoefficient(indexTable[this->m_map.phi1(it)], aij) ;
		}
		nlCoefficient(indexTable[d], -aii) ;
		nlEnd(NL_ROW) ;

		return false ;
	}
} ;

/*******************************************************************************
 * LAPLACIAN COTAN MATRIX : right-hand-side SCALAR
 *******************************************************************************/

template<typename PFP, typename ATTR_TYPE>
class FunctorLaplacianCotanRHS_Scalar : public FunctorMap<typename PFP::MAP>
{
protected:
	const VertexAttribute<unsigned int>& indexTable ;
	const EdgeAttribute<typename PFP::REAL>& edgeWeight ;
	const VertexAttribute<typename PFP::REAL>& vertexArea ;
	const VertexAttribute<ATTR_TYPE>& attrTable ;

public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::REAL REAL ;

	FunctorLaplacianCotanRHS_Scalar(
		MAP& m,
		const VertexAttribute<unsigned int>& index,
		const EdgeAttribute<typename PFP::REAL>& eWeight,
		const VertexAttribute<typename PFP::REAL>& vArea,
		const VertexAttribute<ATTR_TYPE>& attr
	) :	FunctorMap<MAP>(m), indexTable(index), edgeWeight(eWeight), vertexArea(vArea), attrTable(attr)
	{}

	bool operator()(Dart d)
	{
		std::vector<Coeff<REAL> > coeffs ;
		coeffs.reserve(12) ;

		REAL vArea = vertexArea[d] ;
		REAL norm2 = 0 ;
		REAL aii = 0 ;
		Traversor2VE<typename PFP::MAP> t(this->m_map, d) ;
		for(Dart it = t.begin(); it != t.end(); it = t.next())
		{
			REAL aij = edgeWeight[it] / vArea ;
			aii += aij ;
			coeffs.push_back(Coeff<REAL>(indexTable[this->m_map.phi1(it)], aij)) ;
			norm2 += aij * aij ;
		}
		coeffs.push_back(Coeff<REAL>(indexTable[d], -aii)) ;
		norm2 += aii * aii ;

		nlRowParameterd(NL_RIGHT_HAND_SIDE, attrTable[d] * sqrt(norm2)) ;
		nlBegin(NL_ROW);
		for(unsigned int i = 0; i < coeffs.size(); ++i)
			nlCoefficient(coeffs[i].index, coeffs[i].value) ;
		nlEnd(NL_ROW) ;

		return false ;
	}
} ;

/*******************************************************************************
 * LAPLACIAN COTAN MATRIX : right-hand-side VECTOR + coordinate
 *******************************************************************************/

template<typename PFP, typename ATTR_TYPE>
class FunctorLaplacianCotanRHS_Vector : public FunctorMap<typename PFP::MAP>
{
protected:
	const VertexAttribute<unsigned int>& indexTable ;
	const EdgeAttribute<typename PFP::REAL>& edgeWeight ;
	const VertexAttribute<typename PFP::REAL>& vertexArea ;
	const VertexAttribute<ATTR_TYPE>& attrTable ;
	unsigned int coord ;

public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::REAL REAL ;

	FunctorLaplacianCotanRHS_Vector(
		MAP& m,
		const VertexAttribute<unsigned int>& index,
		const EdgeAttribute<typename PFP::REAL>& eWeight,
		const VertexAttribute<typename PFP::REAL>& vArea,
		const VertexAttribute<ATTR_TYPE>& attr,
		unsigned int c
	) :	FunctorMap<MAP>(m), indexTable(index), edgeWeight(eWeight), vertexArea(vArea), attrTable(attr), coord(c)
	{}

	bool operator()(Dart d)
	{
		std::vector<Coeff<REAL> > coeffs ;
		coeffs.reserve(12) ;

		REAL vArea = vertexArea[d] ;
		REAL norm2 = 0 ;
		REAL aii = 0 ;
		Traversor2VE<typename PFP::MAP> t(this->m_map, d) ;
		for(Dart it = t.begin(); it != t.end(); it = t.next())
		{
			REAL aij = edgeWeight[it] / vArea ;
			aii += aij ;
			coeffs.push_back(Coeff<REAL>(indexTable[this->m_map.phi1(it)], aij)) ;
			norm2 += aij * aij ;
		}
		coeffs.push_back(Coeff<REAL>(indexTable[d], -aii)) ;
		norm2 += aii * aii ;

		nlRowParameterd(NL_RIGHT_HAND_SIDE, (attrTable[d])[coord] * sqrt(norm2)) ;
		nlBegin(NL_ROW);
		for(unsigned int i = 0; i < coeffs.size(); ++i)
			nlCoefficient(coeffs[i].index, coeffs[i].value) ;
		nlEnd(NL_ROW) ;

		return false ;
	}
} ;

} // namespace LinearSolving

} // namespace CGoGN

#endif
