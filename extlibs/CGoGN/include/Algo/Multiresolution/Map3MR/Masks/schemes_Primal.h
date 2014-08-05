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

#ifndef __3MR_SCHEMES_PRIMAL__
#define __3MR_SCHEMES_PRIMAL__

#include <cmath>
#include "Algo/Geometry/centroid.h"

namespace CGoGN
{

namespace Multiresolution
{

class MRScheme
{
public:
	MRScheme() {}
	virtual ~MRScheme() {}
	virtual void operator() () = 0 ;
} ;





/* Loop on Boundary Vertices and SHW04 on Insides Vertices
 *********************************************************************************/
template <typename PFP>
class LoopEvenSubdivisionScheme : public MRScheme
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	LoopEvenSubdivisionScheme(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorV<typename PFP::MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			if(m_map.isBoundaryVertex(d))
			{
				Dart db = m_map.findBoundaryFaceOfVertex(d);

				typename PFP::VEC3 np(0) ;
				unsigned int degree = 0 ;
				Traversor2VVaE<typename PFP::MAP> trav(m_map, db) ;
				for(Dart it = trav.begin(); it != trav.end(); it = trav.next())
				{
					++degree ;
					np += m_position[it] ;
				}
				float tmp = 3.0 + 2.0 * cos(2.0 * M_PI / degree) ;
				float beta = (5.0 / 8.0) - ( tmp * tmp / 64.0 ) ;
				np *= beta / degree ;

				typename PFP::VEC3 vp = m_position[db] ;
				vp *= 1.0 - beta ;

				m_map.incCurrentLevel() ;

				m_position[d] = np + vp ;

				m_map.decCurrentLevel() ;
			}
		}
	}
};

template <typename PFP>
class LoopOddSubdivisionScheme : public MRScheme
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	LoopOddSubdivisionScheme(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorE<typename PFP::MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			if(m_map.isBoundaryEdge(d))
			{
				Dart db = m_map.findBoundaryFaceOfEdge(d);
				typename PFP::VEC3 p = loopOddVertex<PFP>(m_map, m_position, db) ;

				m_map.incCurrentLevel() ;

				Dart oddV = m_map.phi2(db) ;
				m_position[oddV] = p ;

				m_map.decCurrentLevel() ;
			}
			else
			{
				typename PFP::VEC3 p = (m_position[d] + m_position[m_map.phi2(d)]) * typename PFP::REAL(0.5);

				m_map.incCurrentLevel() ;

				Dart midV = m_map.phi2(d) ;
				m_position[midV] = p ;

				m_map.decCurrentLevel() ;
			}
		}
	}
};

template <typename PFP>
class SHW04SubdivisionScheme : public MRScheme
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	SHW04SubdivisionScheme(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorW<typename PFP::MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			if(!m_map.isTetrahedron(d))
			{
				typename PFP::VEC3 p = Algo::Geometry::volumeCentroid<PFP>(m_map, d, m_position);

				m_map.incCurrentLevel() ;

				Dart midV = m_map.phi_1(m_map.phi2(m_map.phi1(d)));
				m_position[midV] = p ;

				m_map.decCurrentLevel() ;
			}
		}

		m_map.incCurrentLevel() ;
		TraversorV<typename PFP::MAP> travV(m_map) ;
		for (Dart d = travV.begin(); d != travV.end(); d = travV.next())
		{
			if(!m_map.isBoundaryVertex(d))
			{
				typename PFP::VEC3 p = typename PFP::VEC3(0);
				unsigned int degree = 0;

				Traversor3VW<typename PFP::MAP> travVW(m_map, d);
				for(Dart dit = travVW.begin() ; dit != travVW.end() ; dit = travVW.next())
				{
					p += SHW04Vertex<PFP>(m_map, m_position, dit);
					++degree;
				}

				p /= degree;

				m_position[d] = p ;
			}
		}
		m_map.decCurrentLevel() ;

	}
};

/* Catmull-clark on Boundary Vertices and MJ96 on Insides Vertices
 *********************************************************************************/
template <typename PFP>
class MJ96VertexSubdivision : public MRScheme
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	MJ96VertexSubdivision(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorV<typename PFP::MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			if(m_map.isBoundaryVertex(d))
			{
				Dart db = m_map.findBoundaryFaceOfVertex(d);

				typename PFP::VEC3 np1(0) ;
				typename PFP::VEC3 np2(0) ;
				unsigned int degree1 = 0 ;
				unsigned int degree2 = 0 ;
				Dart it = db ;
				do
				{
					++degree1 ;
					Dart dd = m_map.phi1(it) ;
					np1 += m_position[dd] ;
					Dart end = m_map.phi_1(it) ;
					dd = m_map.phi1(dd) ;
					do
					{
						++degree2 ;
						np2 += m_position[dd] ;
						dd = m_map.phi1(dd) ;
					} while(dd != end) ;
					it = m_map.phi2(m_map.phi_1(it)) ;
				} while(it != db) ;

				float beta = 3.0 / (2.0 * degree1) ;
				float gamma = 1.0 / (4.0 * degree2) ;
				np1 *= beta / degree1 ;
				np2 *= gamma / degree2 ;

				typename PFP::VEC3 vp = m_position[db] ;
				vp *= 1.0 - beta - gamma ;

				m_map.incCurrentLevel() ;

				m_position[d] = np1 + np2 + vp ;

				m_map.decCurrentLevel() ;

			}
			else
			{
				typename PFP::VEC3 P = m_position[d];

				//vertex points
				typename PFP::VEC3 Cavg = typename PFP::VEC3(0);
				unsigned int degree = 0;
				Traversor3VW<typename PFP::MAP> travVW(m_map, d);
				for(Dart dit = travVW.begin() ; dit != travVW.end() ; dit = travVW.next())
				{
					Cavg += Algo::Geometry::volumeCentroid<PFP>(m_map, dit, m_position);
					++degree;
				}
				Cavg /= degree;

				typename PFP::VEC3 Aavg = typename PFP::VEC3(0);
				degree = 0;
				Traversor3VF<typename PFP::MAP> travVF(m_map, d);
				for(Dart dit = travVF.begin() ; dit != travVF.end() ; dit = travVF.next())
				{
					Aavg += Algo::Geometry::faceCentroid<PFP>(m_map, dit, m_position);
					++degree;
				}
				Aavg /= degree;

				typename PFP::VEC3 Mavg = typename PFP::VEC3(0);
				degree = 0;
				Traversor3VE<typename PFP::MAP> travVE(m_map, d);
				for(Dart dit = travVE.begin() ; dit != travVE.end() ; dit = travVE.next())
				{
					Dart d2 = m_map.phi2(dit);
					Aavg += (m_position[dit] + m_position[d2]) * typename PFP::REAL(0.5);
					++degree;
				}
				Aavg /= degree;

				typename PFP::VEC3 vp = Cavg + Aavg * 3 + Mavg * 3 + P;
				vp /= 8;

				m_map.incCurrentLevel() ;

				m_position[d] = P;//vp;

				m_map.decCurrentLevel() ;
			}
		}
	}
};

template <typename PFP>
class MJ96EdgeSubdivision : public MRScheme
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	MJ96EdgeSubdivision(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorE<typename PFP::MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			if(m_map.isBoundaryEdge(d))
			{
				Dart db = m_map.findBoundaryFaceOfEdge(d);

				Dart d1 = m_map.phi2(db) ;
				Dart d2 = m_map.phi2(d1) ;
				Dart d3 = m_map.phi_1(d1) ;
				Dart d4 = m_map.phi_1(d2) ;
				Dart d5 = m_map.phi1(m_map.phi1(d1)) ;
				Dart d6 = m_map.phi1(m_map.phi1(d2)) ;

				typename PFP::VEC3 p1 = m_position[d1] ;
				typename PFP::VEC3 p2 = m_position[d2] ;
				typename PFP::VEC3 p3 = m_position[d3] ;
				typename PFP::VEC3 p4 = m_position[d4] ;
				typename PFP::VEC3 p5 = m_position[d5] ;
				typename PFP::VEC3 p6 = m_position[d6] ;

				p1 *= 3.0 / 8.0 ;
				p2 *= 3.0 / 8.0 ;
				p3 *= 1.0 / 16.0 ;
				p4 *= 1.0 / 16.0 ;
				p5 *= 1.0 / 16.0 ;
				p6 *= 1.0 / 16.0 ;

				m_map.incCurrentLevel() ;

				Dart midV = m_map.phi2(d);

				m_position[midV] = p1 + p2 + p3 + p4 + p5 + p6 ;

				m_map.decCurrentLevel() ;
			}
			else
			{
				//edge points
				typename PFP::VEC3 Cavg = typename PFP::VEC3(0);
				unsigned int degree = 0;
				Traversor3EW<typename PFP::MAP> travEW(m_map, d);
				for(Dart dit = travEW.begin() ; dit != travEW.end() ; dit = travEW.next())
				{
					Cavg += Algo::Geometry::volumeCentroid<PFP>(m_map, dit, m_position);
					++degree;
				}
				Cavg /= degree;

				typename PFP::VEC3 Aavg = typename PFP::VEC3(0);
				degree = 0;
				Traversor3EF<typename PFP::MAP> travEF(m_map, d);
				for(Dart dit = travEF.begin() ; dit != travEF.end() ; dit = travEF.next())
				{
					Aavg += Algo::Geometry::faceCentroid<PFP>(m_map, dit, m_position);
					++degree;
				}
				Aavg /= degree;

				Dart d2 = m_map.phi2(d);
				typename PFP::VEC3 M = (m_position[d] + m_position[d2]) * typename PFP::REAL(0.5);

				typename PFP::VEC3 ep = Cavg + Aavg * 2 + M * (degree - 3);
				ep /= degree;

				m_map.incCurrentLevel() ;

				Dart midV = m_map.phi2(d);

				m_position[midV] = ep;

				m_map.decCurrentLevel() ;
			}
		}
	}
};

template <typename PFP>
class MJ96FaceSubdivision : public MRScheme
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	MJ96FaceSubdivision(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorF<typename PFP::MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			if(m_map.isBoundaryFace(d))
			{
				Dart db = m_map.phi3(d);

				typename PFP::VEC3 p(0) ;
				unsigned int degree = 0 ;
				Traversor2FV<typename PFP::MAP> trav(m_map, db) ;
				for(Dart it = trav.begin(); it != trav.end(); it = trav.next())
				{
					++degree ;
					p += m_position[it] ;
				}
				p /= degree ;

				m_map.incCurrentLevel() ;

				Dart df = m_map.phi1(m_map.phi1(d)) ;

				m_position[df] = p ;

				m_map.decCurrentLevel() ;
			}
			else
			{
				//face points
				typename PFP::VEC3 C0 = Algo::Geometry::volumeCentroid<PFP>(m_map, d, m_position);
				typename PFP::VEC3 C1 = Algo::Geometry::volumeCentroid<PFP>(m_map, m_map.phi3(d), m_position);

				typename PFP::VEC3 A = Algo::Geometry::faceCentroid<PFP>(m_map, m_map.phi3(d), m_position);

				typename PFP::VEC3 fp = C0 + A * 2 + C1;
				fp /= 4;

				m_map.incCurrentLevel() ;

				Dart df = m_map.phi1(m_map.phi1(d)) ;
				m_position[df] = fp;

				m_map.decCurrentLevel() ;
			}
		}
	}
};

template <typename PFP>
class MJ96VolumeSubdivision : public MRScheme
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	MJ96VolumeSubdivision(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorW<typename PFP::MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			//cell points : these points are the average of the
			//vertices of the lattice that bound the cell
			typename PFP::VEC3 p = Algo::Geometry::volumeCentroid<PFP>(m_map, d, m_position);

			m_map.incCurrentLevel() ;

			if(!m_map.isTetrahedron(d))
			{
				Dart midV = m_map.phi_1(m_map.phi2(m_map.phi1(d)));
				m_position[midV] = p ;
			}

			m_map.decCurrentLevel() ;

		}
	}
};

/* Lerp on Boundary Vertices and on Insides Vertices and BSXW02 Averaging
 *********************************************************************************/

template <typename PFP>
class BSXW02AveragingSubdivision : public MRScheme
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	BSXW02AveragingSubdivision(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		m_map.incCurrentLevel() ;
		TraversorV<typename PFP::MAP> trav(m_map) ;
		for (Dart ditE = trav.begin(); ditE != trav.end(); ditE = trav.next())
		{
			if(m_map.isBoundaryVertex(ditE))
			{
				Dart db = m_map.findBoundaryFaceOfVertex(ditE);

				typename PFP::VEC3 P(0);
				unsigned int count = 0;
				Traversor2VF<typename PFP::MAP> travVF(m_map, db);
				for (Dart ditVF = travVF.begin(); ditVF != travVF.end(); ditVF = travVF.next())
				{
					P += Algo::Geometry::faceCentroid<PFP>(m_map, ditVF, m_position);
					++count;
				}

				P /= count;

				m_position[db] = P;
			}
			else if(m_map.isBoundaryEdge(ditE))
			{
				Dart db = m_map.findBoundaryEdgeOfVertex(ditE);

				typename PFP::VEC3 P(0);
				unsigned int count = 0;
				Traversor2VF<typename PFP::MAP> travVF(m_map, db);
				for (Dart ditVF = travVF.begin(); ditVF != travVF.end(); ditVF = travVF.next())
				{
					P += Algo::Geometry::faceCentroid<PFP>(m_map, ditVF, m_position);
					++count;
				}

				P /= count;

				m_position[db] = P;
			}
			else
			{
				typename PFP::VEC3 P(0);
				unsigned int count = 0;
				Traversor3VW<typename PFP::MAP> travVF(m_map, ditE);
				for (Dart ditVF = travVF.begin(); ditVF != travVF.end(); ditVF = travVF.next())
				{
					P += Algo::Geometry::volumeCentroid<PFP>(m_map, ditVF, m_position);
					++count;
				}

				P /= count;

				m_position[ditE] = P;
			}
		}
		m_map.decCurrentLevel() ;
	}
};

template <typename PFP>
class BSXW02EdgeAveragingSubdivision : public MRScheme
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	BSXW02EdgeAveragingSubdivision(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorE<typename PFP::MAP> trav(m_map) ;
		for (Dart ditE = trav.begin(); ditE != trav.end(); ditE = trav.next())
		{
			if(m_map.isBoundaryEdge(ditE))
			{
				Dart db = m_map.findBoundaryFaceOfEdge(ditE);

				m_map.incCurrentLevel() ;

				db = m_map.phi1(db);

				typename PFP::VEC3 P(0);
				unsigned int count = 0;
				Traversor2VF<typename PFP::MAP> travVF(m_map, db);
				for (Dart ditVF = travVF.begin(); ditVF != travVF.end(); ditVF = travVF.next())
				{
					P += Algo::Geometry::faceCentroid<PFP>(m_map, ditVF, m_position);
					++count;
				}

				P /= count;

				m_position[db] = P;

				m_map.decCurrentLevel() ;
			}
		}
	}
};

template <typename PFP>
class BSXW02FaceAveragingSubdivision : public MRScheme
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	BSXW02FaceAveragingSubdivision(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorF<typename PFP::MAP> trav(m_map) ;
		for (Dart ditE = trav.begin(); ditE != trav.end(); ditE = trav.next())
		{
			if(m_map.isBoundaryFace(ditE))
			{
				Dart db = m_map.phi3(ditE);

				m_map.incCurrentLevel() ;

				if(m_map.faceDegree(db) != 3)
				{
					db = m_map.phi2(m_map.phi1(db));

					typename PFP::VEC3 P(0);
					unsigned int count = 0;
					Traversor2VF<typename PFP::MAP> travVF(m_map, db);
					for (Dart ditVF = travVF.begin(); ditVF != travVF.end(); ditVF = travVF.next())
					{
						P += Algo::Geometry::faceCentroid<PFP>(m_map, ditVF, m_position);
						++count;
					}

					P /= count;

					m_position[db] = P;

				}

				m_map.decCurrentLevel() ;
			}
		}
	}
};

template <typename PFP>
class BSXW02VolumeAveragingSubdivision : public MRScheme
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	BSXW02VolumeAveragingSubdivision(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorW<typename PFP::MAP> trav(m_map) ;
		for (Dart ditE = trav.begin(); ditE != trav.end(); ditE = trav.next())
		{
			m_map.incCurrentLevel() ;
			if(!m_map.isTetrahedron(ditE))
			{
				Dart midV = m_map.phi_1(m_map.phi2(m_map.phi1(ditE)));

				typename PFP::VEC3 P(0);
				unsigned int count = 0;
				Traversor3VW<typename PFP::MAP> travVF(m_map, midV);
				for (Dart ditVF = travVF.begin(); ditVF != travVF.end(); ditVF = travVF.next())
				{
					P += Algo::Geometry::volumeCentroid<PFP>(m_map, ditVF, m_position);
					++count;
				}

				P /= count;

				m_position[midV] = P;
			}
			m_map.decCurrentLevel() ;
		}
	}
};

/* DHL93 on Boundary Vertices and MCQ04 on Insides Vertices
 *********************************************************************************/
template <typename PFP>
class  MCQ04VertexSubdivision: public MRScheme
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	MCQ04VertexSubdivision(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		TraversorV<typename PFP::MAP> trav(m_map) ;
		for (Dart d = trav.begin(); d != trav.end(); d = trav.next())
		{
			typename PFP::VEC3 p = m_position[d];

			m_map.incCurrentLevel() ;
			m_position[d] = p ;
			m_map.decCurrentLevel() ;
		}
	}
} ;

template <typename PFP>
class MCQ04EdgeSubdivision : public MRScheme
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	MCQ04EdgeSubdivision(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		typename PFP::REAL omega = 1.0/16.0;

		TraversorE<typename PFP::MAP> trav(m_map) ;
		for (Dart ditE = trav.begin(); ditE != trav.end(); ditE = trav.next())
		{
			if(m_map.isBoundaryEdge(ditE))
			{
				Dart db = m_map.findBoundaryFaceOfEdge(ditE);


				typename PFP::VEC3 p = (m_position[db] + m_position[m_map.phi2(db)]) * typename PFP::REAL(0.5);

				m_map.incCurrentLevel() ;

				Dart midV = m_map.phi2(db) ;
				m_position[midV] = p ;

				m_map.decCurrentLevel() ;

			}
			else
			{
				typename PFP::VEC3 P = ( m_position[ditE] + m_position[m_map.phi2(ditE)] ) * typename PFP::REAL(0.5);

				typename PFP::VEC3 Q(0);
				typename PFP::VEC3 R(0);
				unsigned int count = 0;
				Dart dit = ditE;
				do
				{
					Dart d_1 = m_map.phi_1(dit);
					Dart d11 = m_map.phi1(m_map.phi1(dit));

					Q += m_position[d_1];
					Q += m_position[d11];
					++count;

					Dart dr1 = m_map.phi1(m_map.phi1(m_map.alpha2(d_1)));
					R += m_position[dr1];

					Dart dr2 = m_map.phi1(m_map.phi1(m_map.alpha2(m_map.phi1(dit))));
					R += m_position[dr2];


					dit = m_map.alpha2(dit);
				}while(dit != ditE);

				Q *= (omega / count);

				R *= (omega / count);

				m_map.incCurrentLevel() ;

				Dart midV = m_map.phi2(ditE);

				m_position[midV] = P + Q - R;

				m_map.decCurrentLevel() ;


			}
		}
	}
};

template <typename PFP>
class MCQ04FaceSubdivision : public MRScheme
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	MCQ04FaceSubdivision(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		typename PFP::REAL omega = 1.0/16.0;

		TraversorF<typename PFP::MAP> trav(m_map) ;
		for (Dart ditF = trav.begin(); ditF != trav.end(); ditF = trav.next())
		{
			if(m_map.isBoundaryFace(ditF))
			{
				typename PFP::VEC3 p = Algo::Geometry::faceCentroid<PFP>(m_map, ditF, m_position);

				m_map.incCurrentLevel() ;
				if(m_map.faceDegree(ditF) != 3)
				{
					Dart midF = m_map.phi2(m_map.phi1(ditF));
					m_position[midF] = p ;
				}
				m_map.decCurrentLevel() ;

			}
			else
			{
				//Calcul des Pi
				typename PFP::VEC3 P(0);
				CellMarker<VERTEX> mv(m_map);
				Traversor3FV<typename PFP::MAP> travFV(m_map, ditF);
				for (Dart ditFV = travFV.begin(); ditFV != travFV.end(); ditFV = travFV.next())
				{
					P += m_position[ditFV];
					mv.mark(ditFV);
				}

				P *= (2.0 * omega + 1) / 4.0;

				//Calcul des Qi
				typename PFP::VEC3 Q(0);
				Traversor3FW<typename PFP::MAP> travFW(m_map, ditF);
				for (Dart ditFW = travFW.begin(); ditFW != travFW.end(); ditFW = travFW.next())
				{
					Traversor3WV<typename PFP::MAP> travWV(m_map, ditFW);
					for(Dart ditFV = travWV.begin() ; ditFV != travWV.end() ; ditFV = travWV.next())
					{
						if(!mv.isMarked(ditFV))
						{
							Q += m_position[ditFV];
							mv.mark(ditFV);
						}
					}
				}

				Q *= omega / 4.0;

				//Calcul des Ri
				typename PFP::VEC3 R(0);
				Traversor3FFaE<typename PFP::MAP> travFFaE(m_map, ditF);
				for (Dart ditFFaE = travFFaE.begin(); ditFFaE != travFFaE.end(); ditFFaE = travFFaE.next())
				{
					Traversor3FV<typename PFP::MAP> travFV(m_map, ditFFaE);
					for (Dart ditFV = travFV.begin(); ditFV != travFV.end(); ditFV = travFV.next())
					{
						if(!mv.isMarked(ditFV))
						{
							R += m_position[ditFV];
							mv.mark(ditFV);
						}
					}
				}

				R *= omega / 4.0;

				//Calcul des Si
				typename PFP::VEC3 S(0);
				Traversor3FFaV<typename PFP::MAP> travFFaV(m_map, ditF);
				for (Dart ditFFaV = travFFaV.begin(); ditFFaV != travFFaV.end(); ditFFaV = travFFaV.next())
				{
					Traversor3FV<typename PFP::MAP> travFV(m_map, ditFFaV);
					for (Dart ditFV = travFV.begin(); ditFV != travFV.end(); ditFV = travFV.next())
					{
						if(!mv.isMarked(ditFV))
						{
							S += m_position[ditFV];
							mv.mark(ditFV);
						}
					}
				}

				S *= omega / 8.0;

				m_map.incCurrentLevel() ;
				if(m_map.faceDegree(ditF) != 3)
				{
					Dart midF = m_map.phi2(m_map.phi1(ditF));
					m_position[midF] = P + Q - R - S ;
				}
				m_map.decCurrentLevel() ;
			}
		}
	}
};

template <typename PFP>
class MCQ04VolumeSubdivision : public MRScheme
{
protected:
	typename PFP::MAP& m_map ;
	VertexAttribute<typename PFP::VEC3>& m_position ;

public:
	MCQ04VolumeSubdivision(typename PFP::MAP& m, VertexAttribute<typename PFP::VEC3>& p) : m_map(m), m_position(p)
	{}

	void operator() ()
	{
		typename PFP::REAL omega = 1.0/16.0;

		TraversorW<typename PFP::MAP> trav(m_map) ;
		for (Dart ditW = trav.begin(); ditW != trav.end(); ditW = trav.next())
		{

			if(m_map.isBoundaryAdjacentVolume(ditW))
			{
				typename PFP::VEC3 p = Algo::Geometry::volumeCentroid<PFP>(m_map, ditW, m_position);

				m_map.incCurrentLevel() ;

				if(!m_map.isTetrahedron(ditW))
				{
					Dart midV = m_map.phi_1(m_map.phi2(m_map.phi1(ditW)));
					m_position[midV] = p ;
				}

				m_map.decCurrentLevel() ;
			}
			else
			{
				CellMarker<VERTEX> mv(m_map);

				typename PFP::VEC3 P(0);
				Traversor3WV<typename PFP::MAP> travWV(m_map, ditW);
				for(Dart ditWV = travWV.begin() ; ditWV != travWV.end() ; ditWV = travWV.next())
				{
					P += m_position[ditWV];
					mv.mark(ditWV);
				}

				P *= ((6.0 * omega + 1.0) / 8.0);

				typename PFP::VEC3 Q(0);
				Traversor3WWaF<typename PFP::MAP> travWWaF(m_map, ditW);
				for(Dart ditWWaF = travWV.begin() ; ditWWaF != travWV.end() ; ditWWaF = travWV.next())
				{
					Traversor3WV<typename PFP::MAP> travWV(m_map, ditWWaF);
					for(Dart ditWV = travWV.begin() ; ditWV != travWV.end() ; ditWV = travWV.next())
					{
						if(!mv.isMarked(ditWV))
						{
							Q += m_position[ditWV];
						}
					}
				}

				Q *= omega / 4.0;


				m_map.incCurrentLevel() ;

				if(!m_map.isTetrahedron(ditW))
				{
					Dart midV = m_map.phi_1(m_map.phi2(m_map.phi1(ditW)));
					m_position[midV] = P - Q;
				}

				m_map.decCurrentLevel() ;
			}
		}
	}
};

} // namespace Multiresolution

} // namespace CGoGN


#endif /* __3MR_SCHEMES_PRIMAL__ */
