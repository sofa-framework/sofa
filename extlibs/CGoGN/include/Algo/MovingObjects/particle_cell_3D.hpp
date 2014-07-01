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

// #define DEBUG

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace MovingObjects
{

#define DELTA 0.00001
//static const float DELTA=0.00001;

template <typename PFP>
void ParticleCell3D<PFP>::display()
{
//	std::cout << "position : " << this->m_position << std::endl;
}

template <typename PFP>
typename PFP::VEC3 ParticleCell3D<PFP>::pointInFace(Dart d)
{
	return Algo::Surface::Geometry::faceCentroid<PFP>(m,d,position);
//	const VEC3& p1(this->m_positions[d]);
//	Dart dd=m.phi1(d);
//	const VEC3& p2(this->m_positions[dd]);
//	dd=m.phi1(dd);
//	VEC3& p3(this->m_positions[dd]);
//
//	while(Geom::testOrientation2D(p3,p1,p2)==Geom::ALIGNED) {
//		dd = m.phi1(dd);
//		p3 = this->m_positions[dd];
//	}
//
//	CGoGNout << "pointInFace " << (p1+p3)*0.5f << CGoGNendl;
//
//	return (p1+p3)*0.5f;
}

template <typename PFP>
Geom::Orientation3D ParticleCell3D<PFP>::isLeftENextVertex(VEC3 c, Dart d, VEC3 base)
{
	VEC3 p = position[d];
	VEC3 p1 = position[m.phi1(m.phi2(d))];

	Geom::Plane3D<typename PFP::REAL> pl(p,base,p1);
	return pl.orient(c);
}

template <typename PFP>
bool ParticleCell3D<PFP>::isRightVertex(VEC3 c, Dart d, VEC3 base)
{
	VEC3 p = position[d];
	VEC3 p1 = position[m.phi_1(d)];

	Geom::Plane3D<typename PFP::REAL> pl(p,base,p1);
	return pl.orient(c)==Geom::UNDER;
}

template <typename PFP>
Geom::Orientation3D ParticleCell3D<PFP>::whichSideOfFace(VEC3 c, Dart d)
{
	Geom::Plane3D<typename PFP::REAL> pl = Algo::Surface::Geometry::facePlane<PFP>(m,d,position);
	return pl.orient(c);
}

template <typename PFP>
Geom::Orientation3D ParticleCell3D<PFP>::isLeftL1DVol(VEC3 c, Dart d, VEC3 base, VEC3 top)
{
	VEC3 p2 = position[d];

	Geom::Plane3D<typename PFP::REAL> pl(top,p2,base);

	return pl.orient(c);
}

template <typename PFP>
Geom::Orientation3D ParticleCell3D<PFP>::isRightDVol(VEC3 c, Dart d, VEC3 base, VEC3 top)
{
	VEC3 p1= position[m.phi1(d)];

	Geom::Plane3D<typename PFP::REAL> pl(top,p1,base);

	return pl.orient(c);
}

template <typename PFP>
Geom::Orientation3D ParticleCell3D<PFP>::isAbove(VEC3 c, Dart d, VEC3 top)
{
	VEC3 p1 = position[d];
	VEC3 p2 = position[m.phi1(d)];

	Geom::Plane3D<typename PFP::REAL> pl(top,p2,p1);


	return pl.orient(c);
}

template <typename PFP>
int ParticleCell3D<PFP>::isLeftL1DFace(VEC3 c, Dart d, VEC3 base, VEC3 normal)
{
	VEC3 p2 = position[d];

	VEC3 v2(p2-base);

	VEC3 np = normal ^ v2;

	Geom::Plane3D<typename PFP::REAL> pl(np,base*np);

	return pl.orient(c);
}

template <typename PFP>
bool ParticleCell3D<PFP>::isRightDFace(VEC3 c, Dart d, VEC3 base, VEC3 normal)
{
	VEC3 p1 = position[m.phi1(d)];

	VEC3 np = normal ^ VEC3(p1-base);

	Geom::Plane3D<typename PFP::REAL> pl(np,base*np);

	return pl.orient(c)==Geom::UNDER;
}

template <typename PFP>
Dart ParticleCell3D<PFP>::nextDartOfVertexNotMarked(Dart d, CellMarkerStore<MAP, FACE>& mark)
{
	// lock a marker
	Dart d1;
	DartMarkerNoUnmark<MAP> markCC(m);

	// init algo with parameter dart
	std::list<Dart> darts_list;
	darts_list.push_back(d);
	markCC.mark(d);

	// use iterator for begin of not yet treated darts
	std::list<Dart>::iterator beg = darts_list.begin();

	// until all darts treated
	while (beg != darts_list.end())
	{
		d1 = *beg;
		// add phi1, phi2 and phi3 successor if they are not yet marked
		Dart d2 = m.phi1(m.phi2(d1));
		Dart d3 = m.phi1(m.phi3(d1));

		if (!markCC.isMarked(d2)) {
			darts_list.push_back(d2);
			markCC.mark(d2);
		}

		if (!markCC.isMarked(d3)) {
			darts_list.push_back(d3);
			markCC.mark(d3);
		}

		beg++;

		// apply functor
		if (!mark.isMarked(d1)) {
			for (std::list<Dart>::iterator it=darts_list.begin(); it!=darts_list.end(); ++it)
				markCC.unmark(*it);
			return d1;
		}
	}

	// clear markers
	for (std::list<Dart>::iterator it=darts_list.begin(); it!=darts_list.end(); ++it)
		markCC.unmark(*it);

	return d;
}

template <typename PFP>
Dart ParticleCell3D<PFP>::nextNonPlanar(Dart d)
{
	// lock a marker
	Dart d1;
	DartMarkerNoUnmark<MAP> markCC(m);

	// init algo with parameter dart
	std::list<Dart> darts_list;
	darts_list.push_back(d);
	markCC.mark(d);

	// use iterator for begin of not yet treated darts
	std::list<Dart>::iterator beg = darts_list.begin();

	// until all darts treated
	while (beg != darts_list.end()) {
		d1 = *beg;
		// add phi1, phi2 and phi3 successor if they are not yet marked
		Dart d2 = m.phi1(d1);
		Dart d3 = m.phi2(d1);
		Dart d4 = m.phi_1(d1);

		if (!markCC.isMarked(d2)) {
			darts_list.push_back(d2);
			markCC.mark(d2);
		}

		if (!markCC.isMarked(d3)) {
			darts_list.push_back(d3);
			markCC.mark(d3);
		}

		if (!markCC.isMarked(d4)) {
			darts_list.push_back(d4);
			markCC.mark(d4);
		}

		beg++;

		// apply functor
		Geom::Plane3D<typename PFP::REAL> pl1 = Algo::Surface::Geometry::facePlane<PFP>(m,d,position);
		Geom::Plane3D<typename PFP::REAL> pl2 = Algo::Surface::Geometry::facePlane<PFP>(m,d1,position);
		if ((pl1.normal()-pl2.normal()).norm2()>0.000001)
			beg = darts_list.end();
		// remove the head of the list
	}
	// clear markers
	for (std::list<Dart>::iterator it=darts_list.begin(); it!=darts_list.end(); ++it)
		markCC.unmark(*it);

	return d1;
}

template <typename PFP>
Dart ParticleCell3D<PFP>::nextFaceNotMarked(Dart d, CellMarkerStore<MAP, FACE>& mark)
{
	// lock a marker
	Dart d1;
	DartMarkerNoUnmark<MAP> markCC(m);

	// init algo with parameter dart
	std::list<Dart> darts_list;
	darts_list.push_back(d);
	markCC.mark(d);

	// use iterator for begin of not yet treated darts
	std::list<Dart>::iterator beg = darts_list.begin();

	// until all darts treated
	while (beg != darts_list.end())
	{
		d1 = *beg;

		if(!mark.isMarked(d1))
		{
			for (std::list<Dart>::iterator it=darts_list.begin(); it!=darts_list.end(); ++it)
			{
				markCC.unmark(*it);
			}

			return d1;
		}

		// add phi1, phi2 and phi3 successor if they are not yet marked
		Dart d2 = m.phi1(d1);
		Dart d3 = m.phi2(d1);
		Dart d4 = m.phi_1(d1);

		if (!markCC.isMarked(d2))
		{
			darts_list.push_back(d2);
			markCC.mark(d2);
		}
		if (!markCC.isMarked(d3))
		{
			darts_list.push_back(d3);
			markCC.mark(d3);
		}
		if (!markCC.isMarked(d4))
		{
			darts_list.push_back(d4);
			markCC.mark(d4);
		}
		beg++;
	}

	// clear markers
	for (std::list<Dart>::iterator it=darts_list.begin(); it!=darts_list.end(); ++it)
	{
		markCC.unmark(*it);
	}

//	if(beg==darts_list.end())
		return d;

//	return d1;
}

template <typename PFP>
Geom::Orientation3D ParticleCell3D<PFP>::whichSideOfEdge(VEC3 c, Dart d)
{
	VEC3 p1 = position[m.phi1(d)];
	VEC3 p2 = position[d];

	Geom::Plane3D<typename PFP::REAL> pl = Algo::Surface::Geometry::facePlane<PFP>(m,d,position);
	VEC3 norm = pl.normal();
	VEC3 n2 = norm ^ VEC3(p2-p1);

	Geom::Plane3D<typename PFP::REAL> pl2(n2,p1*n2);

	return pl2.orient(c);
}

template <typename PFP>
bool ParticleCell3D<PFP>::isOnHalfEdge(VEC3 c, Dart d)
{
	VEC3 p1 = position[d];
	VEC3 p2 = position[m.phi1(d)];

	VEC3 norm(p2-p1);
	norm.normalize();

	Geom::Plane3D<typename PFP::REAL> pl(norm,p1*norm);

	return pl.orient(c)==Geom::OVER && !Geom::arePointsEquals(c,p1);
}

/**when the ParticleCell3D trajectory go through a vertex
*  searching the good volume "umbrella" where the ParticleCell3D is
*  if the ParticleCell3D is on the vertex, do nothing */
template <typename PFP>
void ParticleCell3D<PFP>::vertexState(const VEC3& current)
{
	#ifdef DEBUG
	std::cout << "vertexState" << d << std::endl;
	#endif

	crossCell = CROSS_OTHER ;

	VEC3 som = position[d];

	if(Geom::arePointsEquals(current, this->m_position)) {
		this->m_position = this->m_positionFace = som;
		state = VERTEX;
		return;
	}

	Dart dd=d;
	Geom::Orientation3D wsof;
	CellMarkerStore<MAP, FACE> mark(m);

	do {
		VEC3 dualsp = (som+ Algo::Surface::Geometry::vertexNormal<PFP>(m,d,position));
		Dart ddd=d;

		mark.mark(d);

		//searching the good orientation in a volume
		if(isLeftENextVertex(current,d,dualsp)!=Geom::UNDER) {

			d=m.phi1(m.phi2(d));
			while(isLeftENextVertex(current,d,dualsp)!=Geom::UNDER && ddd!=d)
				d=m.phi1(m.phi2(d));

			if(ddd==d) {
				if(isnan(current[0]) || isnan(current[1]) || isnan(current[2])) {
					std::cout << __FILE__ << " " << __LINE__ << " NaN !" << std::endl;
				}

				bool verif = true;
				do {
					if(whichSideOfFace(current,d)!=Geom::OVER)
						verif = false;
					else
						d=m.alpha1(d);
				} while(verif && d!=ddd);

				if(verif) {
					volumeState(current);
					return;
				}
			}
		}
		else {
			while(isRightVertex(current,d,dualsp) && dd!=m.alpha_1(d))
				d=m.phi2(m.phi_1(d));
		}

		wsof = whichSideOfFace(current,d);

		//if c is before the vertex on the edge, we have to change of umbrella, we are symetric to the good umbrella
		if(wsof != Geom::OVER)
		{
			VEC3 p1=position[d];
			VEC3 p2=position[m.phi1(d)];
			VEC3 norm(p2-p1);
			norm.normalize();
			Geom::Plane3D<typename PFP::REAL> plane(norm,p1*norm);

			wsof = plane.orient(current);
		}

		//if c is on the other side of the face, we have to change umbrella
		//if the umbrella has already been tested, we just take another arbitrary one
		if(wsof == 1)
		{
			mark.mark(d);

			if(!mark.isMarked(m.alpha1(d)))
				d=m.alpha1(d);
			else
			{
				Dart dtmp=d;
				d = nextDartOfVertexNotMarked(d,mark);
				if(dtmp==d) {
					std::cout << "numerical rounding ?" << std::endl;

					d = dd;
					this->m_position = pointInFace(d);
					this->m_positionFace = this->m_position;
					volumeState(current);
					return;
				}
			}
		}
	} while(wsof == 1);

	if(wsof != 0)
	{
		this->m_position = pointInFace(d);
		d = nextNonPlanar(d);
		this->m_positionFace = pointInFace(d);
		volumeState(current);
	}
	else
	{
		//Dart ddd=d;
		edgeState(current);
	}
}

template <typename PFP>
void ParticleCell3D<PFP>::edgeState(const VEC3& current)
{
	#ifdef DEBUG
	std::cout << "edgeState" <<  d <<  " " << this->m_position << std::endl;

	#endif

	crossCell = CROSS_OTHER ;

	bool onEdge=false;
	Dart dd=d;
	Geom::Orientation3D wsof = whichSideOfFace(current,m.alpha2(d));
	if(wsof!=Geom::UNDER) {
		do {
			d = m.alpha2(d);
			wsof = whichSideOfFace(current,m.alpha2(d));
		}while(wsof!=1 && dd!=d);

		if(d==dd)
			onEdge = true;

		if(wsof==Geom::ON) {
			switch(whichSideOfEdge(current,d)) {
			case Geom::ON :
				 onEdge=true;
					break;
			case Geom::UNDER :
					d = m.phi2(d);
					break;
			default :
				break;
			}
		}

		wsof = whichSideOfFace(current,d);
	}
	else {
		wsof = whichSideOfFace(current,d);

		while(wsof==Geom::UNDER && dd != m.alpha_2(d)) {
			d = m.alpha_2(d);
			wsof = whichSideOfFace(current,d);
		}

		switch(whichSideOfEdge(current,d)) {
			case Geom::ON : onEdge=true;
					break;
			default :
					break;
		}
	}

	if(wsof==-1)  {

		this->m_position = pointInFace(d);
		d = nextNonPlanar(m.phi1(d));
		this->m_positionFace = pointInFace(d);
		volumeState(current);
		return;
	}
	else {
		if(onEdge) {
			if(isOnHalfEdge(current,d))
				if (isOnHalfEdge(current,m.phi3(d))) {
					state=2;
					this->m_position = this->m_positionFace = current;
				}
				else {
					d=m.phi1(d);
					vertexState(current);
				}
			else {
				vertexState(current);
			}
		}
		else {
			this->m_positionFace = this->m_position;
			d=m.phi1(d);
			faceState(current,wsof);
		}
	}
}

 template <typename PFP>
 void ParticleCell3D<PFP>::faceState(const VEC3& current, Geom::Orientation3D wsof)
{
	#ifdef DEBUG
	std::cout << "faceState" <<  d << std::endl;
	#endif

	crossCell = CROSS_FACE ;

	if(wsof==Geom::ON)
		wsof = whichSideOfFace(current,d);

	if (wsof==Geom::OVER) {
		d = m.phi3(d);
		d = nextNonPlanar(d);
		this->m_positionFace = pointInFace(d);
		volumeState(current);
		return;
	}
	else if(wsof==Geom::UNDER) {
		d = nextNonPlanar(d);
		this->m_positionFace = pointInFace(d);
		volumeState(current);
		return;
	}

	VEC3 norm = Algo::Surface::Geometry::faceNormal<PFP>(m,d,position);

	Dart dd=d;
	if(isLeftL1DFace(current,d,this->m_positionFace,norm)!=Geom::UNDER) {
		d = m.phi_1(d);
		while(isLeftL1DFace(current,d,this->m_positionFace,norm)!=Geom::UNDER && dd!=d)
			d = m.phi_1(d);

		if(dd==d) {
			std::cout << "sortie ?(1)" << std::endl;
			do {
				switch (whichSideOfEdge(current,d)) {
				case Geom::OVER : d=m.phi_1(d);
					break;
				case Geom::ON :this->m_position = current;
					state = EDGE;
					return;
				default :
					Geom::Plane3D<typename PFP::REAL> pl = Algo::Surface::Geometry::facePlane<PFP>(m,d,position);
					VEC3 p3 = pl.normal()+this->m_position;
					Geom::Plane3D<typename PFP::REAL> plOrtho(this->m_position,current,p3);
					VEC3 e(position[m.phi1(d)]-position[d]);

					Geom::intersectionPlaneRay(plOrtho,this->m_position,current-this->m_position,this->m_position);

					edgeState(current);
					return;
				}
			} while(d!=dd);

			this->m_position = this->m_positionFace = current;
			state = FACE;
			return;
		}
	}
	else {
		while(isRightDFace(current,d,this->m_positionFace,norm) && m.phi1(d)!=dd)
			d = m.phi1(d);

		if(m.phi_1(d)==dd) {
			d = m.phi1(d);
			do {
				switch (whichSideOfEdge(current,d))
				{
				case Geom::OVER :
					d=m.phi_1(d);
					break;
				case Geom::ON :this->m_position = current;
					state = EDGE;
					return;
				default :
					 Geom::Plane3D<typename PFP::REAL> pl = Algo::Surface::Geometry::facePlane<PFP>(m,d,position);
					 VEC3 p3 = pl.normal()+this->m_position;
					 Geom::Plane3D<typename PFP::REAL> plOrtho(this->m_position,current,p3);
					 VEC3 e(position[m.phi1(d)]-position[d]);

					 Geom::intersectionPlaneRay(plOrtho,this->m_position,current-this->m_position,this->m_position);
					edgeState(current);
					return;
				}
			}while(d!=dd);

			this->m_position = this->m_positionFace = current;
			state = FACE;
			return;
		}
	}

	switch (whichSideOfEdge(current,d))
	{
	case Geom::OVER :
		 this->m_position = this->m_positionFace = current;
		 state = FACE;
		 break;
	case Geom::ON :
		 this->m_position = this->m_positionFace = current;
		 state = EDGE;
		 break;
	default :
		 Geom::Plane3D<typename PFP::REAL> pl = Algo::Surface::Geometry::facePlane<PFP>(m,d,position);
		 VEC3 p3 = pl.normal()+this->m_position;
		 Geom::Plane3D<typename PFP::REAL> plOrtho(this->m_position,current,p3);
		 VEC3 e(position[m.phi1(d)]-position[d]);

		 Geom::intersectionPlaneRay(plOrtho,this->m_position,current-this->m_position,this->m_position);

		 this->m_positionFace = this->m_position;

		 edgeState(current);
	}
}

template <typename PFP>
void ParticleCell3D<PFP>::volumeState(const VEC3& current)
{
	#ifdef DEBUG
	std::cout << "volumeState " <<  d << std::endl;
	#endif

	CellMarkerStore<MAP, FACE> mark(m);
	bool above;

	Geom::Orientation3D testRight=Geom::OVER;

	do {
		above=false;

		Dart dd=d;
		bool particularcase=false;
		Geom::Orientation3D testLeft = isLeftL1DVol(current,d,this->m_positionFace,this->m_position);

		if(testLeft!=Geom::UNDER) {

			d = m.phi_1(d);

			while(dd!=d && isLeftL1DVol(current,d,this->m_positionFace,this->m_position)!=Geom::UNDER)
				d = m.phi_1(d);

			if(dd==d)
				particularcase=true;
		}
		else {

			testRight = isRightDVol(current,d,this->m_positionFace,this->m_position);


			while(testRight!=Geom::OVER && dd!=m.phi1(d)) {
				d = m.phi1(d);
				testRight = isRightDVol(current,d,this->m_positionFace,this->m_position);
			}

			if(testLeft==0 && dd==m.phi1(d))
				particularcase=true;
		}

		if(particularcase) //(this->m_position,this->m_positionFace,c) presque alignés et si c est proche de la face
				  //aucun des "above" sur les dart ne va donner de résultats concluant (proche de 0 pour tous)
		{
			if(isnan(current[0]) || isnan(current[1]) || isnan(current[2]))
			{
				std::cout << __FILE__ << " " << __LINE__ << " NaN !" << std::endl;
				display();
				this->m_position = current;
				return;
			}

			volumeSpecialCase(current);
			return;
		}

		Geom::Orientation3D testAbove = isAbove(current,d,this->m_position);

		if(testAbove!=Geom::UNDER || (testRight==Geom::ON && isAbove(current,m.phi_1(d),this->m_position)!=Geom::UNDER)) {

			if(testAbove==Geom::OVER || whichSideOfFace(current,d)==Geom::UNDER) {

				mark.mark(d);

				above=true;
				d = m.phi2(d);

				if(mark.isMarked(d)) {
					dd = d;
					d = nextFaceNotMarked(d,mark);
					mark.mark(d);

					if(d==dd) {
						volumeSpecialCase(current);
						return;
					}
				}

				this->m_positionFace = pointInFace(d);
			}
		}
	} while(above);

	Geom::Orientation3D wsof = whichSideOfFace(current,d);

	if(wsof==Geom::UNDER) {
		this->m_position = current;
		state = VOLUME;
	}
	else if(wsof==Geom::ON) {
		if(isAbove(current,d,this->m_position)==Geom::UNDER) {
			this->m_position = this->m_positionFace = current;
			state = FACE;
		}
		else {
			this->m_position = this->m_positionFace = current;
			edgeState(current);
		}
	}
	else {
		if(isAbove(current,d,this->m_position)==Geom::UNDER) {
//			this->m_position = m.intersectDartPlaneLine(d,this->m_position,current);
			Algo::Surface::Geometry::intersectionLineConvexFace<PFP>(m,d,position,this->m_position,current-this->m_position,this->m_position);
			faceState(current,wsof);
		}
		else {
//			this->m_position = m.intersectDartPlaneLineEdge(d,this->m_position,current);
			Algo::Surface::Geometry::intersectionLineConvexFace<PFP>(m,d,position,this->m_position,current-this->m_position,this->m_position);
			edgeState(current);
		}
	}
}

template <typename PFP>
void ParticleCell3D<PFP>::volumeSpecialCase(const VEC3& current)
{
	#ifdef DEBUG
	std::cout << "volumeSpecialCase " <<  d << std::endl;
	#endif

	Dart dd;
	CellMarkerStore<MAP, FACE> mark(m);

	Dart d_min;

	std::vector<Dart> dart_list;
	std::vector<float> dist_list;

	std::list<Dart> visitedFaces;			// Faces that are traversed
	visitedFaces.push_back(d);				// Start with the face of d
	std::list<Dart>::iterator face;

	// For every face added to the list
	for (face = visitedFaces.begin();face != visitedFaces.end(); ++face)
	{
		if (!mark.isMarked(*face))
		{	// Face has not been visited yet

			Geom::Orientation3D wsof = whichSideOfFace(current,*face);
			if(wsof==Geom::OVER)
			{
				d = *face;

				if(isAbove(current,d,this->m_position)==Geom::UNDER)
				{
//					this->m_position = m.intersectDartPlaneLine(d,this->m_position,current);
					Algo::Surface::Geometry::intersectionLineConvexFace<PFP>(m,d,position,this->m_position,current-this->m_position,this->m_position);
					faceState(current,Geom::OVER);
				}
				else
				{
//					this->m_position = m.intersectDartPlaneLineEdge(d,this->m_position,current);
					Algo::Surface::Geometry::intersectionLineConvexFace<PFP>(m,d,position,this->m_position,current-this->m_position,this->m_position);
					edgeState(current);
				}

				return;
			}
			else if(wsof==Geom::ON)
			{
				this->m_position = current;
				d = *face;

				faceState(current);
				return;
			}

			Geom::Plane3D<typename PFP::REAL> pl = Algo::Surface::Geometry::facePlane<PFP>(m,*face,position);
			if(pl.normal()*VEC3(current-this->m_position)>0)
			{
				dist_list.push_back(-pl.distance(current));
				dart_list.push_back(*face);
			}

			// If the face wasn't crossed then mark visited darts (current face)
			// and add non visited adjacent faces to the list of face
			Dart ddd = *face;
			do {
					mark.mark(ddd);
					Dart adj = m.phi2(ddd);			// Get adjacent face
					if (adj != ddd && !mark.isMarked(adj))
						visitedFaces.push_back(adj);// Add it
			} while(ddd!=*face);
		}
	}

	if(dist_list.size()>0) {
		float min=dist_list[0];
		for(unsigned int i = 1;i<dist_list.size();++i)
		{
			if(dist_list[i]<min)
			{
				d=dart_list[i];
				min = dist_list[i];
			}
		}
	}

	this->m_positionFace = pointInFace(d);

	Geom::Orientation3D wsof = whichSideOfFace(current,d);

	if(wsof==Geom::UNDER) {
		this->m_position = current;
		state = VOLUME;
	}
	else if(wsof==Geom::ON) {
		if(isAbove(current,d,this->m_position)==Geom::UNDER) {
			this->m_position = current;
			state = FACE;
		}
		else {
			this->m_position = current;
			state = EDGE;
		}
	}
	else {
		if(isAbove(current,d,this->m_position)==Geom::UNDER)
		{
//			this->m_position = m.intersectDartPlaneLine(d,this->m_position,current);
			Algo::Surface::Geometry::intersectionLineConvexFace<PFP>(m,d,position,this->m_position,current-this->m_position,this->m_position);
			faceState(current,wsof);
		}
		else {
//			this->m_position = m.intersectDartPlaneLineEdge(d,this->m_position,current);
			Algo::Surface::Geometry::intersectionLineConvexFace<PFP>(m,d,position,this->m_position,current-this->m_position,this->m_position);
			edgeState(current);
		}
	}
}

} // namespace MovingObjects

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
