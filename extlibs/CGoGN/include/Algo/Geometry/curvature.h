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

#ifndef __ALGO_GEOMETRY_CURVATURE_H__
#define __ALGO_GEOMETRY_CURVATURE_H__

#include "Geometry/basic.h"

#include "Algo/Selection/collector.h"

#include "Utils/convertType.h"

#include "NL/nl.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

//typedef CPULinearSolverTraits< SparseMatrix<double>, FullVector<double> > CPUSolverTraits ;

template <typename PFP>
void computeCurvatureVertices_QuadraticFitting(
	typename PFP::MAP& map,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin) ;

template <typename PFP>
void computeCurvatureVertex_QuadraticFitting(
	typename PFP::MAP& map,
	Dart dart,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin) ;

template <typename PFP>
void vertexQuadraticFitting(
	typename PFP::MAP& map,
	Dart dart,
	typename PFP::MATRIX33& localFrame,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	float& a, float& b, float& c, float& d, float& e) ;

template <typename PFP>
void quadraticFittingAddVertexPos(typename PFP::VEC3& v, typename PFP::VEC3& p, typename PFP::MATRIX33& localFrame) ;

template <typename PFP>
void quadraticFittingAddVertexNormal(typename PFP::VEC3& v, typename PFP::VEC3& n, typename PFP::VEC3& p, typename PFP::MATRIX33& localFrame) ;
/*
template <typename PFP>
void vertexCubicFitting(Dart dart, typename PFP::VEC3& normal, float& a, float& b, float& c, float& d, float& e, float& f, float& g, float& h, float& i) ;

template <typename PFP>
void cubicFittingAddVertexPos(typename PFP::VEC3& v, typename PFP::VEC3& p, typename PFP::MATRIX33& localFrame) ;

template <typename PFP>
void cubicFittingAddVertexNormal(typename PFP::VEC3& v, typename PFP::VEC3& n, typename PFP::VEC3& p, typename PFP::MATRIX33& localFrame) ;
*/

/* normal cycles by [ACDLD03] : useful for parallel computing*/
template <typename PFP>
void computeCurvatureVertices_NormalCycles(
	typename PFP::MAP& map,
	typename PFP::REAL radius,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	const EdgeAttribute<typename PFP::REAL>& edgeangle,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin,
	VertexAttribute<typename PFP::VEC3>& Knormal,
	unsigned int thread = 0) ;

template <typename PFP>
void computeCurvatureVertex_NormalCycles(
	typename PFP::MAP& map,
	Dart dart,
	typename PFP::REAL radius,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	const EdgeAttribute<typename PFP::REAL>& edgeangle,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin,
	VertexAttribute<typename PFP::VEC3>& Knormal,
	unsigned int thread = 0) ;

template <typename PFP>
void normalCycles_SortAndSetEigenComponents(
	const typename PFP::VEC3& e_val,
	const Geom::Matrix<3,3,typename PFP::REAL> & e_vec,
	typename PFP::REAL& kmax,
	typename PFP::REAL& kmin,
	typename PFP::VEC3& Kmax,
	typename PFP::VEC3& Kmin,
	typename PFP::VEC3& Knormal,
	const typename PFP::VEC3& normal,
	unsigned int thread = 0) ;

template <typename PFP>
void normalCycles_SortTensor( Geom::Matrix<3,3,typename PFP::REAL> & tensor, unsigned int thread = 0) ;

template <typename PFP>
void normalCycles_ProjectTensor( Geom::Matrix<3,3,typename PFP::REAL> & tensor, const typename PFP::VEC3& normal_vector, unsigned int thread = 0) ;

template <typename PFP>
void computeCurvatureVertices_NormalCycles_Projected(
	typename PFP::MAP& map,
	typename PFP::REAL radius,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	const EdgeAttribute<typename PFP::REAL>& edgeangle,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin,
	VertexAttribute<typename PFP::VEC3>& Knormal,
	unsigned int thread = 0) ;

template <typename PFP>
void computeCurvatureVertex_NormalCycles_Projected(
	typename PFP::MAP& map,
	Dart dart,
	typename PFP::REAL radius,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	const EdgeAttribute<typename PFP::REAL>& edgeangle,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin,
	VertexAttribute<typename PFP::VEC3>& Knormal,
	unsigned int thread = 0) ;



/* normal cycles with collector as a parameter : not usable in parallel */

template <typename PFP>
void computeCurvatureVertices_NormalCycles(
	typename PFP::MAP& map,
	Algo::Surface::Selection::Collector<PFP> & neigh,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	const EdgeAttribute<typename PFP::REAL>& edgeangle,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin,
	VertexAttribute<typename PFP::VEC3>& Knormal,
	unsigned int thread = 0) ;

template <typename PFP>
void computeCurvatureVertex_NormalCycles(
	typename PFP::MAP& map,
	Dart dart,
	Algo::Surface::Selection::Collector<PFP> & neigh,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	const EdgeAttribute<typename PFP::REAL>& edgeangle,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin,
	VertexAttribute<typename PFP::VEC3>& Knormal,
	unsigned int thread = 0) ;

template <typename PFP>
void computeCurvatureVertices_NormalCycles_Projected(
	typename PFP::MAP& map,
	Algo::Surface::Selection::Collector<PFP> & neigh,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	const EdgeAttribute<typename PFP::REAL>& edgeangle,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin,
	VertexAttribute<typename PFP::VEC3>& Knormal,
	unsigned int thread = 0) ;

template <typename PFP>
void computeCurvatureVertex_NormalCycles_Projected(
	typename PFP::MAP& map,
	Dart dart,
	Algo::Surface::Selection::Collector<PFP> & neigh,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	const EdgeAttribute<typename PFP::REAL>& edgeangle,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin,
	VertexAttribute<typename PFP::VEC3>& Knormal,
	unsigned int thread = 0) ;


namespace Parallel
{
template <typename PFP>
void computeCurvatureVertices_NormalCycles(
	typename PFP::MAP& map,
	typename PFP::REAL radius,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	const EdgeAttribute<typename PFP::REAL>& edgeangle,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin,
	VertexAttribute<typename PFP::VEC3>& Knormal,
	unsigned int nbth = 0) ;

template <typename PFP>
void computeCurvatureVertices_QuadraticFitting(
	typename PFP::MAP& map,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin,
	unsigned int nbth = 0);

} // namespace Parallel


} // namespace Geometry

}

} // namespace Algo

} // namespace CGoGN

#include "Algo/Geometry/curvature.hpp"

#endif
