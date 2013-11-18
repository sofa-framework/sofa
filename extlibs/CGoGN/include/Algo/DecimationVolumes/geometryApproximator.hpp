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

#include "Algo/DecimationVolumes/selector.h"
#include "Algo/Geometry/centroid.h"

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace Decimation
{

/************************************************************************************
 *							         MID EDGE                                       *
 ************************************************************************************/

template <typename PFP>
bool Approximator_MidEdge<PFP>::init()
{
	return true ;
}

template <typename PFP>
void Approximator_MidEdge<PFP>::approximate(Dart d)
{
	MAP& m = this->m_map ;

	// get some darts
	Dart d1 = m.phi1(d) ;

	// get the contracted edge vertices positions
	VEC3 v1 = this->m_attrV[0]->operator[](d) ;
	VEC3 v2 = this->m_attrV[0]->operator[](d1) ;

	// Compute the approximated position
	this->m_approx[0][d] = (v1 + v2) / REAL(2) ;

	//TODO predictor part
}

/************************************************************************************
 *							         MID FACE                                       *
 ************************************************************************************/

template <typename PFP>
bool Approximator_MidFace<PFP>::init()
{
	return true ;
}

template <typename PFP>
void Approximator_MidFace<PFP>::approximate(Dart d)
{
	MAP& m = this->m_map ;

	// get some darts
	Dart d1 = m.phi1(d) ;
	Dart d_1 = m.phi_1(d) ;

	// get the contracted edge vertices positions
	VEC3 v1 = this->m_attrV[0]->operator[](d) ;
	VEC3 v2 = this->m_attrV[0]->operator[](d1) ;
	VEC3 v3 = this->m_attrV[0]->operator[](d_1) ;

	// Compute the approximated position
	this->m_approx[0][d] = (v1 + v2 + v3) / REAL(3) ;

	//TODO predictor part
}

/************************************************************************************
 *							       	MID VOLUME                                      *
 ************************************************************************************/

template <typename PFP>
bool Approximator_MidVolume<PFP>::init()
{
	return true ;
}

template <typename PFP>
void Approximator_MidVolume<PFP>::approximate(Dart d)
{
	MAP& m = this->m_map ;

	// get some darts
	Dart d1 = m.phi1(d) ;
	Dart d_1 = m.phi_1(d) ;
	Dart d2_1 = m.phi_1(m.phi2(d)) ;

	// get the contracted edge vertices positions
	VEC3 v1 = this->m_attrV[0]->operator[](d) ;
	VEC3 v2 = this->m_attrV[0]->operator[](d1) ;
	VEC3 v3 = this->m_attrV[0]->operator[](d_1) ;
	VEC3 v4 = this->m_attrV[0]->operator[](d2_1) ;

	// Compute the approximated position
	this->m_approx[0][d] = (v1 + v2 + v3 + v4) / REAL(4) ;

	//TODO predictor part
}

/************************************************************************************
 *							       HALF COLLAPSE                                    *
 ************************************************************************************/

template <typename PFP>
bool Approximator_HalfEdgeCollapse<PFP>::init()
{
	return true ;
}

template <typename PFP>
void Approximator_HalfEdgeCollapse<PFP>::approximate(Dart d)
{
	MAP& m = this->m_map ;

	for (unsigned int i = 0 ; i < this->m_attrV.size() ; ++i)
		this->m_approx[i][d] = this->m_attrV[i]->operator[](d) ;

	//TODO predictor part
}

///************************************************************************************
// *                            QUADRIC ERROR METRIC                                  *
// ************************************************************************************/
//template <typename PFP>
//bool Approximator_QEM<PFP>::init()
//{
//	m_quadric = this->m_map.template getAttribute<Utils::Quadric<REAL>, VERTEX>("QEMquadric") ;
//	// Does not require to be valid (if it is not, altenatives will be used).
//
//	if(this->m_predictor)
//	{
//		return false ;
//	}
//	return true ;
//}

} //end namespace Decimation

} //namespace Volume

} //end namespace Algo

} //end namespace CGoGN
