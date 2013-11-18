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
#include "Topology/generic/traversorCell.h"
#include "Topology/generic/traversor2.h"


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
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin)
{
	TraversorV<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		computeCurvatureVertex_QuadraticFitting<PFP>(map, d, position, normal, kmax, kmin, Kmax, Kmin) ;
}

template <typename PFP>
void computeCurvatureVertex_QuadraticFitting(
	typename PFP::MAP& map,
	Dart dart,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin)
{
	typedef typename PFP::REAL REAL ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::MATRIX33 MATRIX33 ;

	VEC3 n = normal[dart] ;

	MATRIX33 localFrame = Algo::Geometry::vertexLocalFrame<PFP>(map, dart, position, n) ;
	MATRIX33 invLocalFrame ;
	localFrame.invert(invLocalFrame) ;

	REAL a, b, c, d, e;
	//vertexCubicFitting(map,dart,localFrame,a,b,c,d,e,f,g,h,i) ;
	vertexQuadraticFitting<PFP>(map, dart, localFrame, position, normal, a, b, c, d, e) ;

	REAL kmax_v, kmin_v, Kmax_x, Kmax_y ;
//	/*int res = */slaev2_(&a, &b, &c, &kmax_v, &kmin_v, &Kmax_x, &Kmax_y) ;

	Eigen::Matrix<REAL,2,2> m;
	m << 2*a, b, b, 2*c;

	// solve eigen problem
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix<REAL,2,2> > solver(m);

	const Eigen::Matrix<REAL,2,1>& ev = solver.eigenvalues();
	kmax_v = ev[0];
	kmin_v = ev[1];

	const Eigen::Matrix<REAL,2,2>& evec = solver.eigenvectors();
	VEC3 Kmax_v(evec(0,0), evec(1,0), 0.0f) ;
	Kmax_v = invLocalFrame * Kmax_v ;
	VEC3 Kmin_v(evec(0,1), evec(1,1), 0.0f) ;
	Kmin_v = invLocalFrame * Kmin_v ;
//	VEC3 Kmin_v = n ^ Kmax_v ;

	if (kmax_v < kmin_v)
	{
		kmax[dart] = -kmax_v ;
		kmin[dart] = -kmin_v ;
		Kmax[dart] = Kmax_v ;
		Kmin[dart] = Kmin_v ;
	}
	else
	{
		kmax[dart] = -kmin_v ;
		kmin[dart] = -kmax_v ;
		Kmax[dart] = Kmin_v ;
		Kmin[dart] = Kmax_v ;
	}
}

template <typename PFP>
void vertexQuadraticFitting(
	typename PFP::MAP& map,
	Dart dart,
	typename PFP::MATRIX33& localFrame,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	float& a, float& b, float& c, float& d, float& e)
{
	typename PFP::VEC3 p = position[dart] ;

	NLContext nlContext = nlNewContext() ;
	nlMakeCurrent(nlContext) ;
	nlSolverParameteri(NL_NB_VARIABLES, 5) ;
	nlSolverParameteri(NL_LEAST_SQUARES, NL_TRUE) ;
	nlBegin(NL_SYSTEM) ;
	nlBegin(NL_MATRIX) ;
	Traversor2VVaE<typename PFP::MAP> tav(map, dart) ;
	for(Dart it = tav.begin(); it != tav.end(); it = tav.next())
	{
		typename PFP::VEC3 v = position[it] ;
		quadraticFittingAddVertexPos<PFP>(v, p, localFrame) ;
		typename PFP::VEC3 n = normal[it] ;
		quadraticFittingAddVertexNormal<PFP>(v, n, p, localFrame) ;
	}
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
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	const EdgeAttribute<typename PFP::REAL>& edgeangle,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin,
	VertexAttribute<typename PFP::VEC3>& Knormal,
	unsigned int thread)
{
	TraversorV<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		computeCurvatureVertex_NormalCycles<PFP>(map, d, radius, position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal,thread) ;
}

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
	unsigned int thread)
{
	typedef typename PFP::REAL REAL ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef Geom::Matrix<3,3,REAL> MATRIX;
	typedef Eigen::Matrix<REAL,3,1> E_VEC3;
	typedef Eigen::Matrix<REAL,3,3,Eigen::RowMajor> E_MATRIX;

	// collect the normal cycle tensor
	Selection::Collector_WithinSphere<PFP> neigh(map, position, radius, thread) ;
	neigh.collectAll(dart) ;

	MATRIX tensor(0) ;
	neigh.computeNormalCyclesTensor(position, edgeangle,tensor);

	// solve eigen problem
	Eigen::SelfAdjointEigenSolver<E_MATRIX> solver (Utils::convertRef<E_MATRIX>(tensor));
	const VEC3& ev = Utils::convertRef<VEC3>(solver.eigenvalues());
	const MATRIX& evec = Utils::convertRef<MATRIX>(solver.eigenvectors());

	normalCycles_SortAndSetEigenComponents<PFP>(ev,evec,kmax[dart],kmin[dart],Kmax[dart],Kmin[dart],Knormal[dart],normal[dart],thread);

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
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	const EdgeAttribute<typename PFP::REAL>& edgeangle,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin,
	VertexAttribute<typename PFP::VEC3>& Knormal,
	unsigned int thread)
{
	TraversorV<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		computeCurvatureVertex_NormalCycles_Projected<PFP>(map, d, radius, position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal,thread) ;
}

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
	unsigned int thread)
{
	typedef typename PFP::REAL REAL ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef Geom::Matrix<3,3,REAL> MATRIX;
	typedef Eigen::Matrix<REAL,3,1> E_VEC3;
	typedef Eigen::Matrix<REAL,3,3,Eigen::RowMajor> E_MATRIX;

	// collect the normal cycle tensor
	Selection::Collector_WithinSphere<PFP> neigh(map, position, radius, thread) ;
	neigh.collectAll(dart) ;

	MATRIX tensor(0) ;
	neigh.computeNormalCyclesTensor(position, edgeangle,tensor);

	// project the tensor
	normalCycles_ProjectTensor<PFP>(tensor,normal[dart],thread);

	// solve eigen problem
	Eigen::SelfAdjointEigenSolver<E_MATRIX> solver (Utils::convertRef<E_MATRIX>(tensor));
	const VEC3& ev = Utils::convertRef<VEC3>(solver.eigenvalues());
	const MATRIX& evec = Utils::convertRef<MATRIX>(solver.eigenvectors());

	normalCycles_SortAndSetEigenComponents<PFP>(ev,evec,kmax[dart],kmin[dart],Kmax[dart],Kmin[dart],Knormal[dart],normal[dart],thread);
}

template <typename PFP>
void computeCurvatureVertices_NormalCycles(
	typename PFP::MAP& map,
	Selection::Collector<PFP> & neigh,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	const EdgeAttribute<typename PFP::REAL>& edgeangle,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin,
	VertexAttribute<typename PFP::VEC3>& Knormal,
	unsigned int thread)
{
	TraversorV<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		computeCurvatureVertex_NormalCycles<PFP>(map, d, neigh, position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal,thread) ;
}

template <typename PFP>
void computeCurvatureVertex_NormalCycles(
	typename PFP::MAP& map,
	Dart dart,
	Selection::Collector<PFP> & neigh,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	const EdgeAttribute<typename PFP::REAL>& edgeangle,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin,
	VertexAttribute<typename PFP::VEC3>& Knormal,
	unsigned int thread)
{
	typedef typename PFP::REAL REAL ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef Geom::Matrix<3,3,REAL> MATRIX;
	typedef Eigen::Matrix<REAL,3,1> E_VEC3;
	typedef Eigen::Matrix<REAL,3,3,Eigen::RowMajor> E_MATRIX;

	// collect the normal cycle tensor
	neigh.collectAll(dart) ;

	MATRIX tensor(0) ;
	neigh.computeNormalCyclesTensor(position, edgeangle,tensor);

	// solve eigen problem
	Eigen::SelfAdjointEigenSolver<E_MATRIX> solver (Utils::convertRef<E_MATRIX>(tensor));
	const VEC3& ev = Utils::convertRef<VEC3>(solver.eigenvalues());
	const MATRIX& evec = Utils::convertRef<MATRIX>(solver.eigenvectors());

	normalCycles_SortAndSetEigenComponents<PFP>(ev,evec,kmax[dart],kmin[dart],Kmax[dart],Kmin[dart],Knormal[dart],normal[dart],thread);
}

template <typename PFP>
void computeCurvatureVertices_NormalCycles_Projected(
	typename PFP::MAP& map,
	Selection::Collector<PFP> & neigh,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	const EdgeAttribute<typename PFP::REAL>& edgeangle,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin,
	VertexAttribute<typename PFP::VEC3>& Knormal,
	unsigned int thread)
{
	TraversorV<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		computeCurvatureVertex_NormalCycles_Projected<PFP>(map, d, neigh, position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal,thread) ;
}

template <typename PFP>
void computeCurvatureVertex_NormalCycles_Projected(
	typename PFP::MAP& map,
	Dart dart,
	Selection::Collector<PFP> & neigh,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	const EdgeAttribute<typename PFP::REAL>& edgeangle,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin,
	VertexAttribute<typename PFP::VEC3>& Knormal,
	unsigned int thread)
{
	typedef typename PFP::REAL REAL ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef Geom::Matrix<3,3,REAL> MATRIX;
	typedef Eigen::Matrix<REAL,3,1> E_VEC3;
	typedef Eigen::Matrix<REAL,3,3,Eigen::RowMajor> E_MATRIX;

	// collect the normal cycle tensor
	neigh.collectAll(dart) ;

	MATRIX tensor(0) ;
	neigh.computeNormalCyclesTensor(position, edgeangle,tensor);

	// project the tensor
	normalCycles_ProjectTensor<PFP>(tensor,normal[dart],thread);

	// solve eigen problem
	Eigen::SelfAdjointEigenSolver<E_MATRIX> solver (Utils::convertRef<E_MATRIX>(tensor));
	const VEC3& ev = Utils::convertRef<VEC3>(solver.eigenvalues());
	const MATRIX& evec = Utils::convertRef<MATRIX>(solver.eigenvectors());

	normalCycles_SortAndSetEigenComponents<PFP>(ev,evec,kmax[dart],kmin[dart],Kmax[dart],Kmin[dart],Knormal[dart],normal[dart],thread);
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
	int inormal=0, imin, imax ;
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
void normalCycles_SortTensor(Geom::Matrix<3,3,typename PFP::REAL> & tensor, unsigned int /*thread*/)
{
	typedef typename PFP::REAL REAL ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef Geom::Matrix<3,3,REAL> MATRIX;
	typedef Eigen::Matrix<REAL,3,1> E_VEC3;
	typedef Eigen::Matrix<REAL,3,3,Eigen::RowMajor> E_MATRIX;

	// compute eigen components
	Eigen::SelfAdjointEigenSolver<E_MATRIX> solver (Utils::convertRef<E_MATRIX>(tensor));
	const VEC3& e_val = Utils::convertRef<VEC3>(solver.eigenvalues());
	const MATRIX& e_vec = Utils::convertRef<MATRIX>(solver.eigenvectors());

	// switch kmin and kmax w.r.t. Kmin and Kmax
	int inormal=0, imin, imax ;
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
void normalCycles_ProjectTensor(Geom::Matrix<3,3,typename PFP::REAL> & tensor, const typename PFP::VEC3& normal_vector, unsigned int thread)
{
	Geom::Matrix<3,3,typename PFP::REAL> proj;
	proj.identity();
	proj -= Geom::transposed_vectors_mult(normal_vector,normal_vector);
	tensor = proj * tensor * proj;
}


namespace Parallel
{

template <typename PFP>
class FunctorComputeCurvatureVertices_NormalCycles: public FunctorMapThreaded<typename PFP::MAP >
{
	typename PFP::REAL m_radius;
	const VertexAttribute<typename PFP::VEC3>& m_position;
	const VertexAttribute<typename PFP::VEC3>& m_normal;
	const EdgeAttribute<typename PFP::REAL>& m_edgeangle;
	VertexAttribute<typename PFP::REAL>& m_kmax;
	VertexAttribute<typename PFP::REAL>& m_kmin;
	VertexAttribute<typename PFP::VEC3>& m_Kmax;
	VertexAttribute<typename PFP::VEC3>& m_Kmin;
	VertexAttribute<typename PFP::VEC3>& m_Knormal;
public:
	 FunctorComputeCurvatureVertices_NormalCycles( typename PFP::MAP& map,
		typename PFP::REAL radius,
		const VertexAttribute<typename PFP::VEC3>& position,
		const VertexAttribute<typename PFP::VEC3>& normal,
		const EdgeAttribute<typename PFP::REAL>& edgeangle,
		VertexAttribute<typename PFP::REAL>& kmax,
		VertexAttribute<typename PFP::REAL>& kmin,
		VertexAttribute<typename PFP::VEC3>& Kmax,
		VertexAttribute<typename PFP::VEC3>& Kmin,
		VertexAttribute<typename PFP::VEC3>& Knormal):
	  FunctorMapThreaded<typename PFP::MAP>(map),
	  m_radius(radius),
	  m_position(position),
	  m_normal(normal),
	  m_edgeangle(edgeangle),
	  m_kmax(kmax),
	  m_kmin(kmin),
	  m_Kmax(Kmax),
	  m_Kmin(Kmin),
	  m_Knormal(Knormal)
	 { }

	void run(Dart d, unsigned int threadID)
	{
		computeCurvatureVertex_NormalCycles<PFP>(this->m_map, d, m_radius, m_position, m_normal, m_edgeangle, m_kmax, m_kmin, m_Kmax, m_Kmin, m_Knormal, threadID) ;
	}
};

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
	unsigned int nbth)
{
	// WAHOO BIG PROBLEM WITH LAZZY EMBEDDING !!!
	if (!map. template isOrbitEmbedded<VERTEX>())
	{
		CellMarkerNoUnmark<VERTEX> cm(map);
		map. template initAllOrbitsEmbedding<VERTEX>();
	}
	if (!map. template isOrbitEmbedded<EDGE>())
	{
		CellMarkerNoUnmark<EDGE> cm(map);
		map. template initAllOrbitsEmbedding<EDGE>();
	}
	if (!map. template isOrbitEmbedded<FACE>())
	{
		CellMarkerNoUnmark<FACE> cm(map);
		map. template initAllOrbitsEmbedding<FACE>();
	}

	FunctorComputeCurvatureVertices_NormalCycles<PFP> funct(map, radius, position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal);
	Algo::Parallel::foreach_cell<typename PFP::MAP,VERTEX>(map, funct, nbth, true);
}



template <typename PFP>
class FunctorComputeCurvatureVertices_QuadraticFitting: public FunctorMapThreaded<typename PFP::MAP >
{
	const VertexAttribute<typename PFP::VEC3>& m_position;
	const VertexAttribute<typename PFP::VEC3>& m_normal;
	VertexAttribute<typename PFP::REAL>& m_kmax;
	VertexAttribute<typename PFP::REAL>& m_kmin;
	VertexAttribute<typename PFP::VEC3>& m_Kmax;
	VertexAttribute<typename PFP::VEC3>& m_Kmin;
public:
	 FunctorComputeCurvatureVertices_QuadraticFitting( typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3>& position,
		const VertexAttribute<typename PFP::VEC3>& normal,
		VertexAttribute<typename PFP::REAL>& kmax,
		VertexAttribute<typename PFP::REAL>& kmin,
		VertexAttribute<typename PFP::VEC3>& Kmax,
		VertexAttribute<typename PFP::VEC3>& Kmin):
	  FunctorMapThreaded<typename PFP::MAP>(map),
	  m_position(position),
	  m_normal(normal),
	  m_kmax(kmax),
	  m_kmin(kmin),
	  m_Kmax(Kmax),
	  m_Kmin(Kmin)
	 { }

	void run(Dart d, unsigned int threadID)
	{
		computeCurvatureVertex_QuadraticFitting<PFP>(this->m_map, d, m_position, m_normal, m_kmax, m_kmin, m_Kmax, m_Kmin) ;
	}
};


template <typename PFP>
void computeCurvatureVertices_QuadraticFitting(
	typename PFP::MAP& map,
	const VertexAttribute<typename PFP::VEC3>& position,
	const VertexAttribute<typename PFP::VEC3>& normal,
	VertexAttribute<typename PFP::REAL>& kmax,
	VertexAttribute<typename PFP::REAL>& kmin,
	VertexAttribute<typename PFP::VEC3>& Kmax,
	VertexAttribute<typename PFP::VEC3>& Kmin,
	unsigned int nbth)
{
	FunctorComputeCurvatureVertices_QuadraticFitting<PFP> funct(map, position, normal, kmax, kmin, Kmax, Kmin);
	Algo::Parallel::foreach_cell<typename PFP::MAP,VERTEX>(map, funct, nbth, true);
}

} // namespace Parallel


} // namespace Geometry

}

} // namespace Algo

} // namespace CGoGN
