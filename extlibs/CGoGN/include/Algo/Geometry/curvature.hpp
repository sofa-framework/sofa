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

#include "Algo/Geometry/localFrame.h"
#include "Geometry/matrix.h"
#include "Topology/generic/traversor/traversorCell.h"
#include "Topology/generic/traversor/traversor2.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

template <typename PFP>
void computeCurvatureVertices_QuadraticFitting(
	typename PFP::MAP& map,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmax,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmax,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmin,
	unsigned int thread)
{

	if ((CGoGN::Parallel::NumberOfThreads > 1) && (thread==0))
	{
		Parallel::computeCurvatureVertices_QuadraticFitting<PFP>(map, position, normal, kmax, kmin, Kmax, Kmin);
		return;
	}

	foreach_cell<VERTEX>(map, [&] (Vertex v)
	{
		computeCurvatureVertex_QuadraticFitting<PFP>(map, v, position, normal, kmax, kmin, Kmax, Kmin) ;
	}
	, FORCE_CELL_MARKING, thread);

//	TraversorV<typename PFP::MAP> t(map) ;
//	for(Vertex v = t.begin(); v != t.end(); v = t.next())
//		computeCurvatureVertex_QuadraticFitting<PFP>(map, v, position, normal, kmax, kmin, Kmax, Kmin) ;
}

template <typename PFP>
void computeCurvatureVertex_QuadraticFitting(
	typename PFP::MAP& map,
	Vertex v,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmax,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmax,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmin)
{
	typedef typename PFP::REAL REAL ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::MATRIX33 MATRIX33 ;

	VEC3 n = normal[v] ;

	MATRIX33 localFrame = Algo::Geometry::vertexLocalFrame<PFP>(map, v, position, n) ;
	MATRIX33 invLocalFrame ;
	localFrame.invert(invLocalFrame) ;

	REAL a, b, c, d, e;
	//vertexCubicFitting(map,dart,localFrame,a,b,c,d,e,f,g,h,i) ;
	vertexQuadraticFitting<PFP>(map, v, localFrame, position, normal, a, b, c, d, e) ;

//	REAL kmax_v, kmin_v, Kmax_x, Kmax_y ;
//	/*int res = */slaev2_(&a, &b, &c, &kmax_v, &kmin_v, &Kmax_x, &Kmax_y) ;

	Eigen::Matrix<REAL,2,2> m;
	m << 2*a, b, b, 2*c;

	// solve eigen problem
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix<REAL,2,2> > solver(m);

	const Eigen::Matrix<REAL,2,1>& ev = solver.eigenvalues();
	REAL kmax_v = ev[0];
	REAL kmin_v = ev[1];

	const Eigen::Matrix<REAL,2,2>& evec = solver.eigenvectors();
	VEC3 Kmax_v(evec(0,0), evec(1,0), 0.0f) ;
	Kmax_v = invLocalFrame * Kmax_v ;
	VEC3 Kmin_v(evec(0,1), evec(1,1), 0.0f) ;
	Kmin_v = invLocalFrame * Kmin_v ;
//	VEC3 Kmin_v = n ^ Kmax_v ;

	if (kmax_v < kmin_v)
	{
		kmax[v] = -kmax_v ;
		kmin[v] = -kmin_v ;
		Kmax[v] = Kmax_v ;
		Kmin[v] = Kmin_v ;
	}
	else
	{
		kmax[v] = -kmin_v ;
		kmin[v] = -kmax_v ;
		Kmax[v] = Kmin_v ;
		Kmin[v] = Kmax_v ;
	}
}

template <typename PFP>
void vertexQuadraticFitting(
	typename PFP::MAP& map,
	Vertex v,
	typename PFP::MATRIX33& localFrame,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal,
	float& a, float& b, float& c, float& d, float& e)
{
	typename PFP::VEC3 p = position[v] ;

	NLContext nlContext = nlNewContext() ;
	nlMakeCurrent(nlContext) ;
	nlSolverParameteri(NL_NB_VARIABLES, 5) ;
	nlSolverParameteri(NL_LEAST_SQUARES, NL_TRUE) ;
	nlBegin(NL_SYSTEM) ;
	nlBegin(NL_MATRIX) ;
	foreach_adjacent2<EDGE>(map, v, [&] (Vertex it) {
		typename PFP::VEC3 itp = position[it] ;
		quadraticFittingAddVertexPos<PFP>(v, itp, localFrame) ;
		typename PFP::VEC3 itn = normal[it] ;
		quadraticFittingAddVertexNormal<PFP>(itp, itn, p, localFrame) ;
	});
	nlEnd(NL_MATRIX) ;
	nlEnd(NL_SYSTEM) ;
	nlSolve() ;

	a = nlGetVariable(0) ;
	b = nlGetVariable(1) ;
	c = nlGetVariable(2) ;
	d = nlGetVariable(3) ;
	e = nlGetVariable(4) ;

	nlDeleteContext(nlContext) ;
}

template <typename PFP>
void quadraticFittingAddVertexPos(typename PFP::VEC3& v, typename PFP::VEC3& p, typename PFP::MATRIX33& localFrame)
{
	typename PFP::VEC3 vec = v - p ;
	vec = localFrame * vec ;

	nlRowParameterd(NL_RIGHT_HAND_SIDE, vec[2]) ;
	nlBegin(NL_ROW) ;
	nlCoefficient(0, vec[0]*vec[0]) ;
	nlCoefficient(1, vec[0]*vec[1]) ;
	nlCoefficient(2, vec[1]*vec[1]) ;
	nlCoefficient(3, vec[0]) ;
	nlCoefficient(4, vec[1]) ;
	nlEnd(NL_ROW) ;
}

template <typename PFP>
void quadraticFittingAddVertexNormal(typename PFP::VEC3& v, typename PFP::VEC3& n, typename PFP::VEC3& p, typename PFP::MATRIX33& localFrame)
{
	typename PFP::VEC3 vec = v - p ;
	vec = localFrame * vec ;
	typename PFP::VEC3 norm = localFrame * n ;

	nlRowParameterd(NL_RIGHT_HAND_SIDE, -1.0f * norm[0]) ;
	nlBegin(NL_ROW);
	nlCoefficient(0, 2.0f * vec[0] * norm[2]) ;
	nlCoefficient(1, vec[1] * norm[2]) ;
	nlCoefficient(2, 0) ;
	nlCoefficient(3, 1.0f * norm[2]) ;
	nlCoefficient(4, 0) ;
	nlEnd(NL_ROW) ;

	nlRowParameterd(NL_RIGHT_HAND_SIDE, -1.0f * norm[1]) ;
	nlBegin(NL_ROW);
	nlCoefficient(0, 0) ;
	nlCoefficient(1, vec[0] * norm[2]) ;
	nlCoefficient(2, 2.0f * vec[1] * norm[2]) ;
	nlCoefficient(3, 0) ;
	nlCoefficient(4, 1.0f * norm[2]) ;
	nlEnd(NL_ROW) ;
}
/*
template <typename PFP>
void vertexCubicFitting(Dart dart, gmtl::Vec3f& normal, float& a, float& b, float& c, float& d, float& e, float& f, float& g, float& h, float& i)
{
	gmtl::Matrix33f localFrame, invLocalFrame ;
	Geometry::vertexLocalFrame<PFP>(m_map,dart,normal,localFrame) ;
	gmtl::invertFull(invLocalFrame, localFrame) ;
	gmtl::Vec3f p = m_map.getVertexEmb(dart)->getPosition() ;
	solverC->reset(false) ;
	solverC->set_least_squares(true) ;
	solverC->begin_system() ;
	Traversor2VVaE<typename PFP::MAP> tav(map, dart) ;
	for(Dart it = tav.begin(); it != tav.end(); it = tav.next())
	{
		// 1-ring vertices
		gmtl::Vec3f v = m_map.getVertexEmb(it)->getPosition() ;
		cubicFittingAddVertexPos(v,p,localFrame) ;
		gmtl::Vec3f n = m_normalsV[m_map.getVertexEmb(it)->getLabel()] ;
		cubicFittingAddVertexNormal(v,n,p,localFrame) ;
	}
	solverC->end_system() ;
	solverC->solve() ;

	a = solverC->variable(0).value() ;
	b = solverC->variable(1).value() ;
	c = solverC->variable(2).value() ;
	d = solverC->variable(3).value() ;
	e = solverC->variable(4).value() ;
	f = solverC->variable(5).value() ;
	g = solverC->variable(6).value() ;
	h = solverC->variable(7).value() ;
	i = solverC->variable(8).value() ;

//	normal = gmtl::Vec3f(-h, -i, 1.0f) ;
//	gmtl::normalize(normal) ;
//	normal = invLocalFrame * normal ;
}

template <typename PFP>
void cubicFittingAddVertexPos(gmtl::Vec3f& v, gmtl::Vec3f& p, gmtl::Matrix33f& localFrame)
{
	gmtl::Vec3f vec = v - p ;
	vec = localFrame * vec ;
	solverC->begin_row() ;

	solverC->add_coefficient(0, vec[0]*vec[0]*vec[0]) ;
	solverC->add_coefficient(1, vec[0]*vec[0]*vec[1]) ;
	solverC->add_coefficient(2, vec[0]*vec[1]*vec[1]) ;
	solverC->add_coefficient(3, vec[1]*vec[1]*vec[1]) ;
	solverC->add_coefficient(4, vec[0]*vec[0]) ;
	solverC->add_coefficient(5, vec[0]*vec[1]) ;
	solverC->add_coefficient(6, vec[1]*vec[1]) ;
	solverC->add_coefficient(7, vec[0]) ;
	solverC->add_coefficient(8, vec[1]) ;

	solverC->set_right_hand_side(vec[2]) ;
	solverC->end_row() ;
}

template <typename PFP>
void cubicFittingAddVertexNormal(gmtl::Vec3f& v, gmtl::Vec3f& n, gmtl::Vec3f& p, gmtl::Matrix33f& localFrame)
{
	gmtl::Vec3f vec = v - p ;
	vec = localFrame * vec ;
	gmtl::Vec3f norm = localFrame * n ;

	solverC->begin_row() ;
	solverC->add_coefficient(0, 3.0f*vec[0]*vec[0]) ;
	solverC->add_coefficient(1, 2.0f*vec[0]*vec[1]) ;
	solverC->add_coefficient(2, vec[1]*vec[1]) ;
	solverC->add_coefficient(3, 0) ;
	solverC->add_coefficient(4, 2.0f*vec[0]) ;
	solverC->add_coefficient(5, vec[1]) ;
	solverC->add_coefficient(6, 0) ;
	solverC->add_coefficient(7, 1.0f) ;
	solverC->add_coefficient(8, 0) ;
	solverC->set_right_hand_side(-1.0f*norm[0]/norm[2]) ;
	solverC->end_row() ;

	solverC->begin_row() ;
	solverC->add_coefficient(0, 0) ;
	solverC->add_coefficient(1, vec[0]*vec[0]) ;
	solverC->add_coefficient(2, 2.0f*vec[0]*vec[1]) ;
	solverC->add_coefficient(3, 3.0f*vec[1]*vec[1]) ;
	solverC->add_coefficient(4, 0) ;
	solverC->add_coefficient(5, vec[0]) ;
	solverC->add_coefficient(6, 2.0f*vec[1]) ;
	solverC->add_coefficient(7, 0) ;
	solverC->add_coefficient(8, 1.0f) ;
	solverC->set_right_hand_side(-1.0f*norm[1]/norm[2]) ;
	solverC->end_row() ;
}
*/

template <typename PFP>
void computeCurvatureVertices_NormalCycles(
	typename PFP::MAP& map,
	typename PFP::REAL radius,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal,
	const EdgeAttribute<typename PFP::REAL, typename PFP::MAP>& edgeangle,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmax,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmax,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Knormal,
	unsigned int thread)
{
	if ((CGoGN::Parallel::NumberOfThreads > 1) && (thread==0))
	{
		Parallel::computeCurvatureVertices_NormalCycles<PFP>(map, radius, position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal);
		return;
	}

	foreach_cell<VERTEX>(map, [&] (Vertex v)
	{
		computeCurvatureVertex_NormalCycles<PFP>(map, v, radius, position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal,thread) ;
	}
	,FORCE_CELL_MARKING,thread);

//	TraversorV<typename PFP::MAP> t(map) ;
//	for(Vertex v = t.begin(); v != t.end(); v = t.next())
//		computeCurvatureVertex_NormalCycles<PFP>(map, v, radius, position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal,thread) ;
}

template <typename PFP>
void computeCurvatureVertex_NormalCycles(
	typename PFP::MAP& map,
	Vertex v,
	typename PFP::REAL radius,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal,
	const EdgeAttribute<typename PFP::REAL, typename PFP::MAP>& edgeangle,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmax,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmax,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Knormal,
	unsigned int thread)
{
	typedef typename PFP::REAL REAL ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef Geom::Matrix<3,3,REAL> MATRIX;
	typedef Eigen::Matrix<REAL,3,1> E_VEC3;
	typedef Eigen::Matrix<REAL,3,3,Eigen::RowMajor> E_MATRIX;

	// collect the normal cycle tensor
	Selection::Collector_WithinSphere<PFP> neigh(map, position, radius, thread) ;
	neigh.collectAll(v) ;

	MATRIX tensor(0) ;
	neigh.computeNormalCyclesTensor(position, edgeangle, tensor);

	// solve eigen problem
	Eigen::SelfAdjointEigenSolver<E_MATRIX> solver(Utils::convertRef<E_MATRIX>(tensor));
	const VEC3& ev = Utils::convertRef<VEC3>(solver.eigenvalues());
	const MATRIX& evec = Utils::convertRef<MATRIX>(solver.eigenvectors());

	normalCycles_SortAndSetEigenComponents<PFP>(ev,evec,kmax[v],kmin[v],Kmax[v],Kmin[v],Knormal[v],normal[v],thread);

//	if (dart.index % 15000 == 0)
//	{
//		CGoGNout << solver.eigenvalues() << CGoGNendl;
//		CGoGNout << solver.eigenvectors() << CGoGNendl;
//		normalCycles_SortTensor<PFP>(tensor);
//		solver.compute(Utils::convertRef<E_MATRIX>(tensor));
//		CGoGNout << solver.eigenvalues() << CGoGNendl;
//		CGoGNout << solver.eigenvectors() << CGoGNendl;
//	}
}

template <typename PFP>
void computeCurvatureVertices_NormalCycles_Projected(
	typename PFP::MAP& map,
	typename PFP::REAL radius,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal,
	const EdgeAttribute<typename PFP::REAL, typename PFP::MAP>& edgeangle,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmax,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmax,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Knormal,
	unsigned int thread)
{
	if ((CGoGN::Parallel::NumberOfThreads > 1) && (thread==0))
	{
		Parallel::computeCurvatureVertices_NormalCycles_Projected<PFP>(map, radius, position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal);
		return;
	}

	foreach_cell<VERTEX>(map, [&] (Vertex v)
	{
		computeCurvatureVertex_NormalCycles_Projected<PFP>(map, v, radius, position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal,thread) ;
	}
	,FORCE_CELL_MARKING,thread);

//	TraversorV<typename PFP::MAP> t(map) ;
//	for(Vertex v = t.begin(); v.dart != t.end(); v = t.next())
//		computeCurvatureVertex_NormalCycles_Projected<PFP>(map, v, radius, position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal, thread) ;
}

template <typename PFP>
void computeCurvatureVertex_NormalCycles_Projected(
	typename PFP::MAP& map,
	Vertex v,
	typename PFP::REAL radius,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal,
	const EdgeAttribute<typename PFP::REAL, typename PFP::MAP>& edgeangle,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmax,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmax,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Knormal,
	unsigned int thread)
{
	typedef typename PFP::REAL REAL ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef Geom::Matrix<3,3,REAL> MATRIX;
	typedef Eigen::Matrix<REAL,3,1> E_VEC3;
	typedef Eigen::Matrix<REAL,3,3,Eigen::RowMajor> E_MATRIX;

	// collect the normal cycle tensor
	Selection::Collector_WithinSphere<PFP> neigh(map, position, radius, thread) ;
	neigh.collectAll(v) ;

	MATRIX tensor(0) ;
	neigh.computeNormalCyclesTensor(position, edgeangle, tensor);

	// project the tensor
	normalCycles_ProjectTensor<PFP>(tensor, normal[v], thread);

	// solve eigen problem
	Eigen::SelfAdjointEigenSolver<E_MATRIX> solver(Utils::convertRef<E_MATRIX>(tensor));
	const VEC3& ev = Utils::convertRef<VEC3>(solver.eigenvalues());
	const MATRIX& evec = Utils::convertRef<MATRIX>(solver.eigenvectors());

	normalCycles_SortAndSetEigenComponents<PFP>(ev,evec,kmax[v],kmin[v],Kmax[v],Kmin[v],Knormal[v],normal[v],thread);
}

template <typename PFP>
void computeCurvatureVertices_NormalCycles(
	typename PFP::MAP& map,
	Algo::Surface::Selection::Collector<PFP>& neigh,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal,
	const EdgeAttribute<typename PFP::REAL, typename PFP::MAP>& edgeangle,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmax,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmax,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Knormal,
	unsigned int thread)
{
	TraversorV<typename PFP::MAP> t(map) ;
	for(Vertex v = t.begin(); v != t.end(); v = t.next())
		computeCurvatureVertex_NormalCycles<PFP>(map, v, neigh, position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal,thread) ;
}

template <typename PFP>
void computeCurvatureVertex_NormalCycles(
	Vertex v,
	Algo::Surface::Selection::Collector<PFP>& neigh,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal,
	const EdgeAttribute<typename PFP::REAL, typename PFP::MAP>& edgeangle,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmax,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmax,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Knormal,
	unsigned int thread)
{
	typedef typename PFP::REAL REAL ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef Geom::Matrix<3,3,REAL> MATRIX;
	typedef Eigen::Matrix<REAL,3,1> E_VEC3;
	typedef Eigen::Matrix<REAL,3,3,Eigen::RowMajor> E_MATRIX;

	// collect the normal cycle tensor
	neigh.collectAll(v) ;

	MATRIX tensor(0) ;
	neigh.computeNormalCyclesTensor(position, edgeangle, tensor);

	// solve eigen problem
	Eigen::SelfAdjointEigenSolver<E_MATRIX> solver(Utils::convertRef<E_MATRIX>(tensor));
	const VEC3& ev = Utils::convertRef<VEC3>(solver.eigenvalues());
	const MATRIX& evec = Utils::convertRef<MATRIX>(solver.eigenvectors());

	normalCycles_SortAndSetEigenComponents<PFP>(ev,evec,kmax[v],kmin[v],Kmax[v],Kmin[v],Knormal[v],normal[v],thread);
}

template <typename PFP>
void computeCurvatureVertices_NormalCycles_Projected(
	typename PFP::MAP& map,
	Algo::Surface::Selection::Collector<PFP>& neigh,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal,
	const EdgeAttribute<typename PFP::REAL, typename PFP::MAP>& edgeangle,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmax,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmax,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Knormal,
	unsigned int thread)
{
	TraversorV<typename PFP::MAP> t(map) ;
	for(Vertex v = t.begin(); v != t.end(); v = t.next())
		computeCurvatureVertex_NormalCycles_Projected<PFP>(map, v, neigh, position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal,thread) ;
}

template <typename PFP>
void computeCurvatureVertex_NormalCycles_Projected(
	Vertex v,
	Algo::Surface::Selection::Collector<PFP>& neigh,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal,
	const EdgeAttribute<typename PFP::REAL, typename PFP::MAP>& edgeangle,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmax,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmax,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Knormal,
	unsigned int thread)
{
	typedef typename PFP::REAL REAL ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef Geom::Matrix<3,3,REAL> MATRIX;
	typedef Eigen::Matrix<REAL,3,1> E_VEC3;
	typedef Eigen::Matrix<REAL,3,3,Eigen::RowMajor> E_MATRIX;

	// collect the normal cycle tensor
	neigh.collectAll(v) ;

	MATRIX tensor(0) ;
	neigh.computeNormalCyclesTensor(position, edgeangle, tensor);

	// project the tensor
	normalCycles_ProjectTensor<PFP>(tensor, normal[v], thread);

	// solve eigen problem
	Eigen::SelfAdjointEigenSolver<E_MATRIX> solver(Utils::convertRef<E_MATRIX>(tensor));
	const VEC3& ev = Utils::convertRef<VEC3>(solver.eigenvalues());
	const MATRIX& evec = Utils::convertRef<MATRIX>(solver.eigenvectors());

	normalCycles_SortAndSetEigenComponents<PFP>(ev,evec,kmax[v],kmin[v],Kmax[v],Kmin[v],Knormal[v],normal[v],thread);
}

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
	unsigned int /*thread*/)
{
	// sort eigen components : ev[inormal] has minimal absolute value ; kmin = ev[imin] <= ev[imax] = kmax
	int inormal = 0, imin, imax ;
	if (fabs(e_val[1]) < fabs(e_val[inormal])) inormal = 1;
	if (fabs(e_val[2]) < fabs(e_val[inormal])) inormal = 2;
	imin = (inormal + 1) % 3;
	imax = (inormal + 2) % 3;
	if (e_val[imax] < e_val[imin]) { int tmp = imin ; imin = imax ; imax = tmp ; }

	// set curvatures from sorted eigen components
	// warning : Kmin and Kmax are switched w.r.t. kmin and kmax
	// normal direction : minimal absolute eigen value
	Knormal[0] = e_vec(0,inormal);
	Knormal[1] = e_vec(1,inormal);
	Knormal[2] = e_vec(2,inormal);
	if (Knormal * normal < 0) Knormal *= -1; // change orientation
	// min curvature
	kmin = e_val[imin] ;
	Kmin[0] = e_vec(0,imax);
	Kmin[1] = e_vec(1,imax);
	Kmin[2] = e_vec(2,imax);
	// max curvature
	kmax = e_val[imax] ;
	Kmax[0] = e_vec(0,imin);
	Kmax[1] = e_vec(1,imin);
	Kmax[2] = e_vec(2,imin);
}

template <typename PFP>
void normalCycles_SortTensor(Geom::Matrix<3,3,typename PFP::REAL>& tensor, unsigned int /*thread*/)
{
	typedef typename PFP::REAL REAL ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef Geom::Matrix<3,3,REAL> MATRIX;
	typedef Eigen::Matrix<REAL,3,1> E_VEC3;
	typedef Eigen::Matrix<REAL,3,3,Eigen::RowMajor> E_MATRIX;

	// compute eigen components
	Eigen::SelfAdjointEigenSolver<E_MATRIX> solver(Utils::convertRef<E_MATRIX>(tensor));
	const VEC3& e_val = Utils::convertRef<VEC3>(solver.eigenvalues());
	const MATRIX& e_vec = Utils::convertRef<MATRIX>(solver.eigenvectors());

	// switch kmin and kmax w.r.t. Kmin and Kmax
	int inormal = 0, imin, imax ;
	if (fabs(e_val[1]) < fabs(e_val[inormal])) inormal = 1;
	if (fabs(e_val[2]) < fabs(e_val[inormal])) inormal = 2;
	imin = (inormal + 1) % 3;
	imax = (inormal + 2) % 3;
	if (e_val[imax] < e_val[imin]) { int tmp = imin ; imin = imax ; imax = tmp ; }

	tensor = e_vec;
	int i; REAL v;
	i = inormal; v = e_val[inormal];
	tensor(0,i) *= v; tensor(1,i) *= v; tensor(2,i) *= v;
	i = imin; v = e_val[imax];
	tensor(0,i) *= v; tensor(1,i) *= v; tensor(2,i) *= v;
	i = imax; v = e_val[imin];
	tensor(0,i) *= v; tensor(1,i) *= v; tensor(2,i) *= v;
	tensor = tensor*e_vec.transposed();
}

template <typename PFP>
void normalCycles_ProjectTensor(Geom::Matrix<3,3,typename PFP::REAL>& tensor, const typename PFP::VEC3& normal_vector, unsigned int thread)
{
	Geom::Matrix<3,3,typename PFP::REAL> proj;
	proj.identity();
	proj -= Geom::transposed_vectors_mult(normal_vector, normal_vector);
	tensor = proj * tensor * proj;
}


namespace Parallel
{

template <typename PFP>
void computeCurvatureVertices_NormalCycles(
	typename PFP::MAP& map,
	typename PFP::REAL radius,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal,
	const EdgeAttribute<typename PFP::REAL, typename PFP::MAP>& edgeangle,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmax,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmax,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Knormal)
{
	// WAHOO BIG PROBLEM WITH LAZZY EMBEDDING !!!
	if (!map.template isOrbitEmbedded<VERTEX>())
		Algo::Topo::initAllOrbitsEmbedding<VERTEX>(map);

	if (!map.template isOrbitEmbedded<EDGE>())
		Algo::Topo::initAllOrbitsEmbedding<EDGE>(map);

	if (!map.template isOrbitEmbedded<FACE>())
		Algo::Topo::initAllOrbitsEmbedding<FACE>(map);

	CGoGN::Parallel::foreach_cell<VERTEX>(map, [&] (Vertex v, unsigned int threadID)
	{
		computeCurvatureVertex_NormalCycles<PFP>(map, v, radius, position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal, threadID) ;
	}, FORCE_CELL_MARKING);
}

template <typename PFP>
void computeCurvatureVertices_NormalCycles_Projected(
	typename PFP::MAP& map,
	typename PFP::REAL radius,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal,
	const EdgeAttribute<typename PFP::REAL, typename PFP::MAP>& edgeangle,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmax,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmax,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Knormal)
{
//	// WAHOO BIG PROBLEM WITH LAZZY EMBEDDING !!!
	if (!map.template isOrbitEmbedded<VERTEX>())
		Algo::Topo::initAllOrbitsEmbedding<VERTEX>(map);

	if (!map.template isOrbitEmbedded<EDGE>())
		Algo::Topo::initAllOrbitsEmbedding<EDGE>(map);

	if (!map.template isOrbitEmbedded<FACE>())
		Algo::Topo::initAllOrbitsEmbedding<FACE>(map);

	CGoGN::Parallel::foreach_cell<VERTEX>(map, [&] (Vertex v, unsigned int threadID)
	{
		computeCurvatureVertex_NormalCycles_Projected<PFP>(map, v, radius, position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal, threadID) ;
	}, FORCE_CELL_MARKING);
}

template <typename PFP>
void computeCurvatureVertices_QuadraticFitting(
	typename PFP::MAP& map,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmax,
	VertexAttribute<typename PFP::REAL, typename PFP::MAP>& kmin,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmax,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& Kmin)
{
	CGoGN::Parallel::foreach_cell<VERTEX>(map, [&] (Vertex v, unsigned int threadID)
	{
		computeCurvatureVertex_QuadraticFitting<PFP>(map, v, position, normal, kmax, kmin, Kmax, Kmin, threadID) ;
	}, FORCE_CELL_MARKING);

}

} // namespace Parallel


} // namespace Geometry

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
