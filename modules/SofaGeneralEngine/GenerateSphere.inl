/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_ENGINE_GENERATESPHERE_INL
#define SOFA_COMPONENT_ENGINE_GENERATESPHERE_INL

#include "GenerateSphere.h"
#include <sofa/helper/rmath.h> //M_PI
#include <sofa/helper/system/config.h>

namespace sofa
{

namespace component
{

namespace engine
{


/*===========================================================
          Definition of the tetrahedron
===========================================================*/



static unsigned int tetrahedron_faces_triangles[4][3]=
{{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
// static unsigned int tetrahedron_faces_neighbor[4][3]=
// {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
static unsigned int tetrahedron_edge_vertex[6][2]=
{{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
// static unsigned int tetrahedron_edge_triangle[6][2]=
// {{2,3},{1,3},{2,1},{0,3},{2,0},{0,1}};


/*===========================================================
          Definition of the Octahedron
===========================================================*/

// static double octahedron_vertices_pos[6][3]=
// {	{0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
// 	{0.0, 0.0, -1.0},{-1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}};
static unsigned int octahedron_faces_triangles[8][3]=
{{0, 1, 2}, {0, 2, 4}, {0, 4, 5}, {0, 5, 1},
 {1, 5, 3}, {1, 3, 2}, {3, 5, 4}, {2, 3, 4}};
// static unsigned int octahedron_faces_neighbor[8][3]=
// {{5,1,3},{7,2,0},{6,3,1},{4,0,2},
//  {6,5,3},{7,0,4},{2,7,4},{6,1,5}};
static unsigned int octahedron_edge_vertex[12][2]=
{{0,1},{0,2},{0,4},{0,5},{1,2},{1,3},
 {1,5},{2,3},{2,4},{3,4},{3,5},{4,5}};
// static unsigned int octahedron_edge_triangle[12][2]=
// {{0,3},{0,1},{1,2},{2,3},{0,5},{4,5},
//  {3,4},{5,7},{1,7},{6,7},{4,6},{2,6}};

/*===========================================================
          Definition of the Icosahedron
===========================================================*/

static double icosahedron_vertices_pos[12][3]=
{{0.4249358858193742, -0.6234212590752522, -0.901521749427252},
 {-0.5999406907417391, 0.05914133178387721, -1.009227188986324},
 {0.4980672603913731, 0.6101449500603583, -0.872707378430782},
 {1.170388006823068, -0.06734654200022822, -0.0873067158600265},
 {0.4878971283264427, -1.037062929459035, 0.2615777778398508},
 {-0.606226178027942, -0.958889124295748, -0.3082004094765843},
 {-0.4249358858193759, 0.6234212590752533, 0.90152174942725},
 {0.5999406907417364, -0.05914133178387813, 1.009227188986327},
 {-0.4980672603913764, -0.6101449500603557, 0.87270737843078},
 {-1.170388006823072, 0.0673465420002311, 0.0873067158600247},
 {-0.4878971283264417, 1.037062929459036, -0.2615777778398516},
 {0.606226178027942, 0.958889124295747, 0.3082004094765856}};
static unsigned int icosahedron_faces_triangles[20][3]=
{{0, 1, 5}, {0, 2, 1}, {0, 3, 2}, {0, 4, 3}, {0, 5, 4},
 {1, 2, 10}, {1, 9, 5}, {1, 10, 9}, {2, 3, 11}, {2, 11, 10},
 {3, 4, 7}, {3, 7, 11}, {4, 5, 8}, {4, 8, 7}, {5, 9, 8},
 {6, 7, 8}, {6, 8, 9}, {6, 9, 10}, {6, 10, 11}, {6, 11, 7}};
// static unsigned int icosahedron_faces_neighbor[20][3]=
// {{6,4,1},{5,0,2},{8,1,3},{10,2,4},{12,3,0},
//  {9,7,1},{14,0,7},{17,6,5},{11,9,2},{18,5,8},
//  {13,11,3},{19,8,10},{14,13,4},{15,10,12},{16,12,6},
//  {13,16,19},{14,17,15},{7,18,16},{9,19,17},{11,15,18}};
static unsigned int icosahedron_edge_vertex[30][2]=
{{0,1},{0,2},{0,3},{0,4},{0,5},
 {1,2},{1,5},{1,9},{1,10},{2,3},
 {2,10},{2,11},{3,4},{3,7},{3,11},
 {4,5},{4,7},{4,8},{5,8},{5,9},
 {6,7},{6,8},{6,9},{6,10},{6,11},
 {7,8},{7,11},{8,9},{9,10},{10,11}};
// static unsigned int icosahedron_edge_triangle[30][2]=
// {{0,1},{1,2},{2,3},{3,4},{0,4},
//  {1,5},{0,6},{6,7},{7,5},{2,8},
//  {5,9},{8,9},{3,10},{10,11},{8,11},
//  {4,12},{10,13},{12,13},{12,14},{6,14},
//  {15,19},{15,16},{16,17},{17,18},{18,19},
//  {13,15},{11,19},{14,16},{7,17},{9,18}};

const unsigned int edgesInTetrahedronArray[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
const unsigned int adjacentVerticesToEdges[6][2] = {{2,3}, {1,3}, {1,2}, {0,3}, {0,2}, {0,1}};

template <class DataTypes>
GenerateSphere<DataTypes>::GenerateSphere()
    : f_outputTetrahedraPositions ( initData (&f_outputTetrahedraPositions, "output_TetrahedraPosition", "output array of 3d points of tetrahedra mesh") )
    , f_tetrahedra( initData (&f_tetrahedra, "tetrahedra", "output mesh tetrahedra") )
	, f_outputTrianglesPositions ( initData (&f_outputTrianglesPositions, "output_TrianglesPosition", "output array of 3d points of triangle mesh") )
	, f_triangles( initData (&f_triangles, "triangles", "output triangular mesh") )
	, f_bezierTetrahedronDegree( initData (&f_bezierTetrahedronDegree, "BezierTetrahedronDegree", "order of Bezier tetrahedra") )
	, f_bezierTetrahedronWeight( initData (&f_bezierTetrahedronWeight, "BezierTetrahedronWeights", "weights of rational Bezier tetrahedra") )
	, f_isBezierTetrahedronRational( initData (&f_isBezierTetrahedronRational, "isBezierTetrahedronRational", "booleans indicating if each Bezier tetrahedron is rational or integral") )
	, f_bezierTriangleDegree( initData (&f_bezierTriangleDegree, "BezierTriangleDegree", "order of Bezier triangles") )
	, f_bezierTriangleWeight( initData (&f_bezierTriangleWeight, "BezierTriangleWeights", "weights of rational Bezier triangles") )
	, f_isBezierTriangleRational( initData (&f_isBezierTriangleRational, "isBezierTriangleRational", "booleans indicating if each Bezier triangle is rational or integral") )
    , f_radius( initData (&f_radius,(Real)0.2, "radius", "input sphere radius") )
	, f_origin( initData (&f_origin,Coord(), "origin", "sphere center point") )
    , f_tessellationDegree( initData (&f_tessellationDegree,(size_t)1, "tessellationDegree", "Degree of tessellation of each Platonic triangulation") )
    , f_platonicSolidName( initData (&f_platonicSolidName,std::string("icosahedron"), "platonicSolid", "name of the Platonic triangulation used to create the spherical dome : either \"tetrahedron\", \"octahedron\" or \"icosahedron\"") )
{
    addAlias(&f_outputTetrahedraPositions,"position");
    addAlias(&f_outputTetrahedraPositions,"output_position");
}


template <class DataTypes>
void GenerateSphere<DataTypes>::init()
{
    addInput(&f_tessellationDegree);
    addInput(&f_origin);
    addInput(&f_platonicSolidName);


	addOutput(&f_triangles);
    addOutput(&f_outputTrianglesPositions);
    addOutput(&f_bezierTriangleWeight);
	addOutput(&f_isBezierTriangleRational);


    addOutput(&f_tetrahedra);
    addOutput(&f_outputTetrahedraPositions);
	addOutput(&f_bezierTetrahedronWeight);
	addOutput(&f_isBezierTetrahedronRational);

    setDirtyValue();

	if (f_platonicSolidName.getValue() == "icosahedron"){
		platonicSolid=ICOSAHEDRON;
	}
	else if (f_platonicSolidName.getValue() == "tetrahedron"){
		platonicSolid=TETRAHEDRON;
	} else if (f_platonicSolidName.getValue() == "octahedron"){
		platonicSolid=OCTAHEDRON;
	} else {
		serr << "Wrong Platonic Solid Name : "<< f_platonicSolidName <<sendl;
		serr << "It should be either \"tetrahedron\", \"octahedron\" or \"icosahedron\" "<<sendl;
	}

}

template <class DataTypes>
void GenerateSphere<DataTypes>::reinit()
{

    update();
}

template <class DataTypes>
void GenerateSphere<DataTypes>::update()
{
    const Real radius = f_radius.getValue();
	const size_t frequency = f_tessellationDegree.getValue();
	const Coord origin = f_origin.getValue();
	const PlatonicTriangulation solid=platonicSolid;


    cleanDirty();

	helper::WriteOnlyAccessor<Data<VecCoord> > posTrian = f_outputTrianglesPositions;
    helper::WriteOnlyAccessor<Data<SeqTriangles> > trians = f_triangles;
    helper::WriteOnlyAccessor<Data<VecCoord> > posTetra = f_outputTetrahedraPositions;
    helper::WriteOnlyAccessor<Data<SeqTetrahedra> > tetras = f_tetrahedra;
	std::vector<Triangle> platonicTriangles;

	size_t  nbVertices;
	size_t  nbTriangles;
	size_t i,j,k,l;
	Coord pos;
	Edge e;

	std::vector<Edge> edgeArray;
	std::vector<Triangle> edgeTriangleArray;
	std::map<Edge,size_t> edgeMap;

	if (solid==TETRAHEDRON) {
		nbVertices=4;
		pos=Coord(0.f,0.f,1.f);posTrian.push_back(pos);
		pos=Coord(0.f,(Real)(2*M_SQRT2/3.),-1.f/3.f);posTrian.push_back(pos);
		pos=Coord((Real)(-M_SQRT2/sqrt(3.)),(Real)( -M_SQRT2/3), -1.f/3.f);posTrian.push_back(pos);
		pos=Coord((Real)(M_SQRT2/sqrt(3.)), (Real)(-M_SQRT2/3), -1.f/3.f);posTrian.push_back(pos);
		// edge array
		for (i=0;i<6;++i) {
			e=Edge(tetrahedron_edge_vertex[i][0],tetrahedron_edge_vertex[i][1]);
			edgeArray.push_back(e);
			edgeMap.insert(std::pair<Edge,size_t>(e,i));
		}
		nbTriangles=4;
		for(i=0;i<4;++i) {
			Triangle tr(tetrahedron_faces_triangles[i][0],tetrahedron_faces_triangles[i][1],tetrahedron_faces_triangles[i][2]);
			platonicTriangles.push_back(tr);
			// fill triangle edge array
		}
	} else if (solid==OCTAHEDRON) {
		nbVertices=6;
		pos=Coord(0,0,0);pos[2]=1.0;posTrian.push_back(pos);
		pos=Coord(0,0,0);pos[0]=1.0;posTrian.push_back(pos);
		pos=Coord(0,0,0);pos[1]=1.0;posTrian.push_back(pos);
		pos=Coord(0,0,0);pos[2]= -1.0;posTrian.push_back(pos);
		pos=Coord(0,0,0);pos[0]= -1.0;posTrian.push_back(pos);
		pos=Coord(0,0,0);pos[1]= -1.0;posTrian.push_back(pos);
		// edge array
		for (i=0;i<12;++i) {
			e=Edge(octahedron_edge_vertex[i][0],octahedron_edge_vertex[i][1]);
			edgeArray.push_back(e);
			edgeMap.insert(std::pair<Edge,size_t>(e,i));
		}
		nbTriangles=8;

		for(i=0;i<8;++i) {
			Triangle tr(octahedron_faces_triangles[i][0],octahedron_faces_triangles[i][1],octahedron_faces_triangles[i][2]);
			platonicTriangles.push_back(tr);
		}
	} else {
		// icosahedron
		nbVertices=12;
		for(i=0;i<12;++i) {
			pos=Coord((Real)icosahedron_vertices_pos[i][0],(Real)icosahedron_vertices_pos[i][1],(Real)icosahedron_vertices_pos[i][2]);
			pos/=pos.norm();
			posTrian.push_back(pos);
		}
		// edge array
		for (i=0;i<30;++i) {
			e=Edge(icosahedron_edge_vertex[i][0],icosahedron_edge_vertex[i][1]);
			edgeArray.push_back(e);
			edgeMap.insert(std::pair<Edge,size_t>(e,i));
		}
		nbTriangles=20;
		for(i=0;i<nbTriangles;++i) {
			Triangle tr(icosahedron_faces_triangles[i][0],icosahedron_faces_triangles[i][1],icosahedron_faces_triangles[i][2]);
			platonicTriangles.push_back(tr);
		}
	}
	// now eventually tessellate each triangle and  project points on the sphere
	if (frequency>1) {

		// first build the triangle to edge array : get edge index for each triangle
		for (i=0;i<platonicTriangles.size();++i) {
			Triangle tr;

			for (j=0;j<3;++j) {
				Edge e=Edge(platonicTriangles[i][(j+1)%3],platonicTriangles[i][(j+2)%3]);
				// sort edge
				Edge se=Edge(std::min(e[0],e[1]),std::max(e[0],e[1]));
				std::map<Edge,size_t>::iterator itm;
				itm=edgeMap.find(se);

				assert(itm!=edgeMap.end());
				tr[j]=(PointID)(*itm).second;
			}
			edgeTriangleArray.push_back(tr);
		}
		// tessellate edges
		Real w,phi;
		Coord normal;
		for (i=0;i<edgeArray.size();++i) {
			normal=defaulttype::cross<Real>(posTrian[edgeArray[i][0]],posTrian[edgeArray[i][1]]);
			normal=defaulttype::cross(normal,posTrian[edgeArray[i][0]]);
			normal/= normal.norm();
			phi=acos(dot(posTrian[edgeArray[i][0]],posTrian[edgeArray[i][1]]));
			for (j=1;j<frequency;++j) {
				// spherical interpolation rather than linear interpolation
				w=(Real) j/(Real) frequency;
				pos=cos(w*phi)*posTrian[edgeArray[i][0]]+
					sin(w*phi)*normal;

//				pos=posTrian[edgeArray[i][0]]*(1-w);
//				pos+=posTrian[edgeArray[i][1]]*(w);
//				pos/=pos.norm();
				posTrian.push_back(pos);
			}
		}

		size_t vertexRank;
		size_t nbVerticesInsideTriangle=(frequency-2)*(frequency-1)/2;
		// create subtriangle array
		//create a temporary map associating Triangle coordinate with its index
		std::map<Triangle,size_t> triangleIndexMap;
		// insert triangle vertex
		triangleIndexMap.insert(std::pair<Triangle,size_t>(Triangle((PointID)frequency,0,0),0));
		triangleIndexMap.insert(std::pair<Triangle,size_t>(Triangle(0,frequency,0),1));
		triangleIndexMap.insert(std::pair<Triangle,size_t>(Triangle(0,0,frequency),2));
		vertexRank=3;
		// insert edge vertex;
		for (j=0;j<3;++j) {
			for (k=1;k<frequency;++k) {
				Triangle tr;
				tr[j]=0;
				tr[(j+1)%3]=frequency-k;
				tr[(j+2)%3]=k;
				triangleIndexMap.insert(std::pair<Triangle,size_t>(tr,vertexRank++));
			}
		}
		// insert triangle vertex
		for (j=1;j<frequency;++j) {
			for (k=1;k<(frequency-j);++k) {
				l=frequency-j-k;
				Triangle tr;
				tr[0]=j;tr[1]=k;tr[2]=l;
				triangleIndexMap.insert(std::pair<Triangle,size_t>(tr,vertexRank++));
			}
		}
		// now create the array subtriangleArray where the frequency*frequency subtriangles are defined
		std::vector<Triangle> subtriangleArray;
		std::map<Triangle,size_t>::iterator omi;
		Triangle tbi[3],tr;
		for ( i=1;i<=frequency;++i) {
			for (size_t j=0;j<(frequency-i+1);++j) {
				tbi[0]=Triangle(i,j,frequency-i-j);
				tbi[1]=Triangle(i-1,j+1,frequency-i-j);
				tbi[2]=Triangle(i-1,j,frequency-i-j+1);
				for (k=0;k<3;++k) {
					omi=triangleIndexMap.find(tbi[k]);
					assert(omi!=triangleIndexMap.end());
					tr[k]= (*omi).second;
				}

				subtriangleArray.push_back(tr);
				if ((i+j)<frequency) {
					tbi[2]=Triangle(i,j+1,frequency-i-j-1);
					tr[2]=tr[1];
					omi=triangleIndexMap.find(tbi[2]);
					assert(omi!=triangleIndexMap.end());

					tr[1]= (*omi).second;
					subtriangleArray.push_back(tr);
				}
			}
		}


		// generate tessellated triangles
		for (i=0;i<platonicTriangles.size();++i) {
			/// create additional points inside triangles
			if (frequency>2) {
				for (j=1;j<frequency;++j) {
					for (k=1;k<(frequency-j);++k) {
						l=frequency-k-j;
						pos=(Real)j*posTrian[platonicTriangles[i][0]]+(Real)k*posTrian[platonicTriangles[i][1]]+
							(Real)l*posTrian[platonicTriangles[i][2]];
						pos/=pos.norm();

						posTrian.push_back(pos);
					}
				}
			}
			// define frequency*frequency subtriangles
			std::vector<size_t> macroTriangle;
			// store vertices
			for(j=0;j<3;++j)
				macroTriangle.push_back(platonicTriangles[i][j]);
			// store edge points
			for(j=0;j<3;++j) {
				Edge e;
				e[0]=platonicTriangles[i][(j+1)%3];
				e[1]=platonicTriangles[i][(j+2)%3];
				vertexRank=nbVertices+(frequency-1)*edgeTriangleArray[i][j];
				if (e[0]<e[1]) {
					for (k=1;k<frequency;++k)
						macroTriangle.push_back(vertexRank+k-1);
				} else {
					for (k=1;k<frequency;++k)
						macroTriangle.push_back(vertexRank+frequency-1-k);
				}
			}
			// store triangle points
			vertexRank=nbVertices+(frequency-1)*edgeArray.size()+i*nbVerticesInsideTriangle;
			for(j=0;j<nbVerticesInsideTriangle;++j)
				macroTriangle.push_back(vertexRank+j);
			// dumps subtriangles
			for (j=0;j<subtriangleArray.size();++j) {
				for(k=0;k<3;++k) {
					tr[k]=macroTriangle[subtriangleArray[j][k]];
				}
				trians.push_back(tr);
			}

		}
	} else {
		trians.resize(platonicTriangles.size());
		std::copy(platonicTriangles.begin(),platonicTriangles.end(),trians.begin());
	}
	// now create tetrahedral mesh
	if ((solid==TETRAHEDRON) && (frequency==1)){
		// specific case : one 1 tetrahedra
		// in this copy the position
		posTetra.resize(posTrian.size());
		std::copy(posTrian.begin(),posTrian.end(),posTetra.begin());
		// add the regular tetrahedron
		tetras.resize(1);
		Tetrahedron tet(0,1,2,3);
		tetras[0]=tet;
	} else {
		/// now create tetrahedra by simply adding the central point
		posTetra.resize(posTrian.size()+1);
		// add the sphere center on tetrahedra
		posTetra[0]=Coord();
		// add triangulation points
		for (i=0;i<posTrian.size();++i)
			posTetra[i+1]=posTrian[i];
		// create tetrahedra by adding the center to each triangle
		tetras.resize(trians.size());
		for (i=0;i<trians.size();++i) {
			Tetrahedron tet(0,trians[i][0]+1,trians[i][1]+1,trians[i][2]+1);
			tetras[i]=tet;
		}
	}
	// now handle bezier triangulation
	if (f_bezierTriangleDegree.getValue()>1) {

		size_t degreeTriangle=f_bezierTriangleDegree.getValue();
		// fill the bezier triangle weight to 1 for regular triangle vertices
		helper::WriteOnlyAccessor<Data <sofa::helper::vector<Real> > > bezierTriangleWeight=f_bezierTriangleWeight;
		// initialize the weight to 1
		bezierTriangleWeight.resize(posTrian.size());
		std::fill(bezierTriangleWeight.begin(),bezierTriangleWeight.end(),(Real)1.0);
		// initialize the rational flag for each triangle to true
		helper::WriteOnlyAccessor<Data <sofa::helper::vector<bool> > >  isRationalSpline=f_isBezierTriangleRational;
		isRationalSpline.resize(nbTriangles);
		// sets the first set of triangles to be rational
		std::fill(isRationalSpline.begin(),isRationalSpline.end(),true);

		// parse edges
		edgeMap.clear();
		std::map<Edge,size_t>::iterator item;
		SeqTriangles::iterator itt;
		Real phi,ctheta;
		Coord posTmp,posTmp2,normal;
		edgeArray.clear();

		for (itt=trians.begin();itt!=trians.end();++itt) {

			for (i=0;i<3;++i){
				Edge e,se;
				e[0]=(*itt)[(i+1)%3];
				e[1]=(*itt)[(i+2)%3];
				if (e[0]>e[1]){
					se[0]=e[1];se[1]=e[0];
				} else {
					se=e;
				}
				if ((item=edgeMap.find(se))==edgeMap.end()){
					edgeMap.insert(std::pair<Edge,size_t>(se,edgeArray.size()));
					edgeArray.push_back(e);
					// add Bezier points along the edge
					phi=acos(dot(posTrian[se[0]],posTrian[se[1]]));
					normal=defaulttype::cross(posTrian[(*itt)[2]]-posTrian[(*itt)[1]],
						posTrian[(*itt)[2]]-posTrian[(*itt)[0]]);
					normal/=normal.norm();
					ctheta=dot(normal,posTrian[(*itt)[1]]);
					// ctheta must be negative otherwise get negative weights
					if (ctheta>0) {
						normal*= -1.0f;
						ctheta*= -1.0f;
					}
					if (degreeTriangle==2) {
						pos=(posTrian[e[0]]+posTrian[e[1]])/2.0;
						pos/=pos.norm();
						pos/=cos(phi/2);
						posTrian.push_back(pos);
						bezierTriangleWeight.push_back((Real)cos(phi/2));
					} else if (degreeTriangle==3) {
						posTmp=(posTrian[e[0]]+posTrian[e[1]])/(1+dot(posTrian[e[0]],posTrian[e[1]]));
						pos=(2*ctheta*posTmp+posTrian[e[0]])/(1+2*ctheta);
						posTrian.push_back(pos);
						pos=(2*ctheta*posTmp+posTrian[e[1]])/(1+2*ctheta);
						posTrian.push_back(pos);
						bezierTriangleWeight.push_back((Real)(1+2*ctheta)/3.0f);
						bezierTriangleWeight.push_back((Real)(1+2*ctheta)/3.0f);
					} else if (degreeTriangle==4) {
						Coord P110=(posTrian[e[0]]+posTrian[e[1]])/(1+dot(posTrian[e[0]],posTrian[e[1]]));
						Real w110=((posTrian[e[0]]+posTrian[e[1]])/2).norm();
						Real w310=(2-ctheta-dot(normal,P110));
						posTmp=(P110*(1-ctheta)+posTrian[e[0]]*(1-dot(normal,P110)))/w310;
						Coord P130=((1-ctheta)*P110+(1-dot(normal,P110))*posTrian[e[1]])/w310;
						w310*=w110/(2*(1-ctheta));
						posTrian.push_back(posTmp);
						bezierTriangleWeight.push_back((Real)w310);
						Real w220=4*(1-dot(normal,P110))*w110*w110+2*(1-ctheta);
						Coord P220=P110*4*(1-dot(normal,P110))*w110*w110+
							(posTrian[e[0]]+posTrian[e[1]])*(1-ctheta);
						P220/=w220;
						w220/=6*(1-ctheta);
						posTrian.push_back(P220);
						bezierTriangleWeight.push_back((Real)w220);
						posTrian.push_back(P130);
						bezierTriangleWeight.push_back((Real)w310);
					} else {
						for (j=1;j<degreeTriangle;++j) {
							// interpolated position
							pos= ((Real) j*posTrian[e[1]]+(Real)(degreeTriangle-j)*posTrian[e[0]])/degreeTriangle;
							posTrian.push_back(pos);
							// weight is 1
							bezierTriangleWeight.push_back((Real)1.0f);
						}
					}
				}
			}
		}
		if (degreeTriangle>2) {

			for (itt=trians.begin();itt!=trians.end();++itt) {
				// add inside points
				if (degreeTriangle==4) {
					normal=defaulttype::cross(posTrian[(*itt)[2]]-posTrian[(*itt)[1]],
						posTrian[(*itt)[2]]-posTrian[(*itt)[0]]);
					normal/=normal.norm();
					ctheta=dot(normal,posTrian[(*itt)[1]]);
					// ctheta must be negative otherwise get negative weights
					if (ctheta>0) {
						normal*= -1.0f;
						ctheta*= -1.0f;
					}
					Coord mid[3];
					Real wmid[3];
					for (i=0;i<3;++i) {
						mid[i]=(posTrian[(*itt)[(i+1)%3]]+posTrian[(*itt)[(i+2)%3]])/(1+dot(posTrian[(*itt)[(i+1)%3]],posTrian[(*itt)[(i+2)%3]]));
						wmid[i]=((posTrian[(*itt)[(i+1)%3]]+posTrian[(*itt)[(i+2)%3]])/2).norm();
					}
					Coord posTmp;
					Real ww;
					for (j=3;j>0;--j) {
						i=(j+2)%3;
						ww= 2*wmid[(i+1)%3]*wmid[(i+2)%3]*(1+dot(mid[(i+1)%3],mid[(i+2)%3])-dot(normal,mid[(i+1)%3])-dot(normal,mid[(i+2)%3]))+
							wmid[i]*(1+dot(posTrian[(*itt)[i]],mid[i])-ctheta-dot(mid[i],normal));
						posTmp=2*wmid[(i+1)%3]*wmid[(i+2)%3]*((1-dot(normal,mid[(i+1)%3]))*mid[(i+2)%3]+(1-dot(normal,mid[(i+2)%3]))*mid[(i+1)%3]+
							(dot(mid[(i+1)%3],mid[(i+2)%3])-1)*normal)+wmid[i]*((1-ctheta)*mid[i]+(1-dot(normal,mid[i]))*posTrian[(*itt)[i]]+(dot(posTrian[(*itt)[i]],mid[i])-1)*normal);
						posTmp/=ww;
						ww/=6*(1-ctheta);
						posTrian.push_back(posTmp);
						bezierTriangleWeight.push_back((Real)ww);
					}
				} else {
					for (i=1;i<(degreeTriangle-1);++i) {
						for (j=1;j<(degreeTriangle-i);++j) {
							pos= ((Real) i*posTrian[(*itt)[0]]+
								(Real)j*posTrian[(*itt)[1]]+(Real) (degreeTriangle-i-j)*posTrian[(*itt)[2]])/degreeTriangle;
							posTrian.push_back(pos);
							bezierTriangleWeight.push_back((Real)1.0f);
						}
					}
				}
			}
		}
	}

		// now handle bezier tetrahedra
	if (f_bezierTetrahedronDegree.getValue()>1) {
		size_t degreeTetrahedron=f_bezierTetrahedronDegree.getValue();
		// fill the bezier triangle weight to 1 for regular tetrahedron vertices
		helper::WriteOnlyAccessor<Data <sofa::helper::vector<Real> > > bezierTetrahedronWeight=f_bezierTetrahedronWeight;
		// initialize the weight to 1
		bezierTetrahedronWeight.resize(posTetra.size());
		std::fill(bezierTetrahedronWeight.begin(),bezierTetrahedronWeight.end(),(Real)1.0);
		// initialize the rational flag for each tetrahedron to true
		helper::WriteOnlyAccessor<Data <sofa::helper::vector<bool> > >  isRationalSpline=f_isBezierTetrahedronRational;
		isRationalSpline.resize(tetras.size());
		// sets the first set of tetrahedra to be rational
		std::fill(isRationalSpline.begin(),isRationalSpline.end(),true);

		// parse edges
		edgeMap.clear();
		std::map<Edge,size_t>::iterator item;
		SeqTetrahedra::iterator itt;
		Real phi,ctheta;
		Coord posTmp,posTmp2,normal;
		edgeArray.clear();
		for (itt=tetras.begin();itt!=tetras.end();++itt) {
		}

		for (itt=tetras.begin();itt!=tetras.end();++itt) {

			for (i=0;i<6;++i){
				Edge e,se;
				e[0]=(*itt)[edgesInTetrahedronArray[i][0]];
				e[1]=(*itt)[edgesInTetrahedronArray[i][1]];
				if (e[0]>e[1]){
					se[0]=e[1];se[1]=e[0];
				} else {
					se=e;
				}
				if ((item=edgeMap.find(se))==edgeMap.end()){
					edgeMap.insert(std::pair<Edge,size_t>(se,edgeArray.size()));
					edgeArray.push_back(e);
					// check if the edge is on the sphere
					bool onSphere=false;
					if ((fabs(posTetra[e[0]].norm2()-1)<1e-4)&& (fabs(posTetra[e[1]].norm2()-1)<1e-4)){
						onSphere=true;

					}
					if (onSphere) {
						// add Bezier points along the edge
						phi=acos(dot(posTetra[e[0]],posTetra[e[1]]));
						// one of the 2 other vertices must be on the sphere
						size_t nextVertexOnSphere=(*itt)[adjacentVerticesToEdges[i][0]];
						if (fabs(posTetra[nextVertexOnSphere].norm2()-1)>1e-4){
							nextVertexOnSphere=(*itt)[adjacentVerticesToEdges[i][1]];
							assert(fabs(posTetra[nextVertexOnSphere].norm2()-1)<1e-4);
						}
						// we have 3 points on the sphere therefore can compute a normal
						normal=defaulttype::cross(posTetra[nextVertexOnSphere]-posTetra[e[1]],
							posTetra[nextVertexOnSphere]-posTetra[e[0]]);
						normal/=normal.norm();
						ctheta=dot(normal,posTetra[e[1]]);
						// ctheta must be negative otherwise get negative weights
						if (ctheta>0) {
							normal*= -1.0f;
							ctheta*= -1.0f;
						}
						if (degreeTetrahedron==2) {
							pos=(posTetra[e[0]]+posTetra[e[1]])/2.0;
							pos/=pos.norm();
							pos/=cos(phi/2);
							posTetra.push_back(pos);
							bezierTetrahedronWeight.push_back((Real)cos(phi/2));
						} else if (degreeTetrahedron==3) {
							posTmp=(posTetra[e[0]]+posTetra[e[1]])/(1+dot(posTetra[e[0]],posTetra[e[1]]));
							pos=(2*ctheta*posTmp+posTetra[se[0]])/(1+2*ctheta);
							posTetra.push_back(pos);
							pos=(2*ctheta*posTmp+posTetra[se[1]])/(1+2*ctheta);
							posTetra.push_back(pos);
							bezierTetrahedronWeight.push_back((Real)(1+2*ctheta)/3.0f);
							bezierTetrahedronWeight.push_back((Real)(1+2*ctheta)/3.0f);
						} else if (degreeTetrahedron==4) {
							Coord P110=(posTetra[e[0]]+posTetra[e[1]])/(1+dot(posTetra[e[0]],posTetra[e[1]]));
							Real w110=((posTetra[e[0]]+posTetra[e[1]])/2).norm();
							Real w310=(2-ctheta-dot(normal,P110));
							posTmp=(P110*(1-ctheta)+posTetra[se[0]]*(1-dot(normal,P110)))/w310;
							Coord P130=((1-ctheta)*P110+(1-dot(normal,P110))*posTetra[se[1]])/w310;
							w310*=w110/(2*(1-ctheta));
							posTetra.push_back(posTmp);
							bezierTetrahedronWeight.push_back((Real)w310);
							Real w220=4*(1-dot(normal,P110))*w110*w110+2*(1-ctheta);
							Coord P220=P110*4*(1-dot(normal,P110))*w110*w110+
								(posTetra[e[0]]+posTetra[e[1]])*(1-ctheta);
							P220/=w220;
							w220/=6*(1-ctheta);
							posTetra.push_back(P220);
							bezierTetrahedronWeight.push_back((Real)w220);
							posTetra.push_back(P130);
							bezierTetrahedronWeight.push_back((Real)w310);
						} else {
							for (j=1;j<degreeTetrahedron;++j) {
								// interpolated position
								pos= ((Real) j*posTetra[se[1]]+(Real)(degreeTetrahedron-j)*posTetra[se[0]])/degreeTetrahedron;
								posTetra.push_back(pos);
								// weight is 1
								bezierTetrahedronWeight.push_back((Real)1.0f);
							}
						}
					} else {
						for (j=1;j<degreeTetrahedron;++j) {
							// interpolated position
							pos= ((Real) j*posTetra[se[1]]+(Real)(degreeTetrahedron-j)*posTetra[se[0]])/degreeTetrahedron;
							posTetra.push_back(pos);
							// weight is 1
							bezierTetrahedronWeight.push_back((Real)1.0f);
						}
					}



				}
			}
		}

		if (degreeTetrahedron>2) {
			std::set<Triangle> triangleSet;
			std::set<Triangle>::iterator itemtr;
			for (itt=tetras.begin();itt!=tetras.end();++itt) {
				for (k=0;k<4;++k){
					// add inside points

					Triangle tr,sortedtr;
					if (k%2)
						tr=Triangle((*itt)[(k+1)%4],(*itt)[(k+2)%4],(*itt)[(k+3)%4]);
					else
						tr=Triangle((*itt)[(k+1)%4],(*itt)[(k+3)%4],(*itt)[(k+2)%4]);
					// permute until small index at tr[0]
					while ((tr[0]>tr[1]) || (tr[0]>tr[2]))
					{
						unsigned int val=tr[0];
						tr[0]=tr[1];
						tr[1]=tr[2];
						tr[2]=val;
					}
					// check if a triangle with an opposite orientation already exists
					sortedtr = Triangle(tr[0], tr[2], tr[1]);

					// finds unique triangle
					if ((itemtr=triangleSet.find(sortedtr))==triangleSet.end()){
						triangleSet.insert(tr);

						if (degreeTetrahedron==4) {

							bool onSphere=true;
							for (i=0;i<3;++i) {
								if (fabs(posTetra[tr[i]].norm2()-1)>1e-4)
									onSphere=false;
							}
							if (onSphere) {
								normal=defaulttype::cross(posTetra[tr[2]]-posTetra[tr[1]],
									posTetra[tr[2]]-posTetra[tr[0]]);
								normal/=normal.norm();
								ctheta=dot(normal,posTetra[tr[1]]);
								// ctheta must be negative otherwise get negative weights
								if (ctheta>0) {
									normal*= -1.0f;
									ctheta*= -1.0f;
								}
								Coord mid[3];
								Real wmid[3];
								for (i=0;i<3;++i) {
									mid[i]=(posTetra[tr[(i+1)%3]]+posTetra[tr[(i+2)%3]])/(1+dot(posTetra[tr[(i+1)%3]],posTetra[tr[(i+2)%3]]));
									wmid[i]=((posTetra[tr[(i+1)%3]]+posTetra[tr[(i+2)%3]])/2).norm();
								}
								Coord posTmp;
								Real ww;
								for (j=3;j>0;--j) {
									i=(j+2)%3;
									ww= 2*wmid[(i+1)%3]*wmid[(i+2)%3]*(1+dot(mid[(i+1)%3],mid[(i+2)%3])-dot(normal,mid[(i+1)%3])-dot(normal,mid[(i+2)%3]))+
										wmid[i]*(1+dot(posTetra[tr[i]],mid[i])-ctheta-dot(mid[i],normal));
									posTmp=2*wmid[(i+1)%3]*wmid[(i+2)%3]*((1-dot(normal,mid[(i+1)%3]))*mid[(i+2)%3]+(1-dot(normal,mid[(i+2)%3]))*mid[(i+1)%3]+
										(dot(mid[(i+1)%3],mid[(i+2)%3])-1)*normal)+wmid[i]*((1-ctheta)*mid[i]+(1-dot(normal,mid[i]))*posTetra[tr[i]]+(dot(posTetra[tr[i]],mid[i])-1)*normal);
									posTmp/=ww;
									ww/=6*(1-ctheta);
									posTetra.push_back(posTmp);
									bezierTetrahedronWeight.push_back((Real)ww);
								}
							} else {
								for (i=1;i<(degreeTetrahedron-1);++i) {
									for (j=1;j<(degreeTetrahedron-i);++j) {
										pos= ((Real) i*posTetra[tr[0]]+
											(Real)j*posTetra[tr[1]]+(Real) (degreeTetrahedron-i-j)*posTetra[tr[2]])/degreeTetrahedron;
										posTetra.push_back(pos);
										bezierTetrahedronWeight.push_back((Real)1.0f);
									}
								}
							}
						} else {
							for (i=1;i<(degreeTetrahedron-1);++i) {
								for (j=1;j<(degreeTetrahedron-i);++j) {
									pos= ((Real) i*posTetra[tr[0]]+
										(Real)j*posTetra[tr[1]]+(Real) (degreeTetrahedron-i-j)*posTetra[tr[2]])/degreeTetrahedron;
									posTetra.push_back(pos);
									bezierTetrahedronWeight.push_back((Real)1.0f);
								}
							}
						}
					}
				}
			}
		}
		if (degreeTetrahedron>3) {
			for (itt=tetras.begin();itt!=tetras.end();++itt) {
				// use default position
				for (i=1;i<(degreeTetrahedron-2);++i) {
					for (j=1;j<(degreeTetrahedron-i-1);++j) {
						for (k=1;k<(degreeTetrahedron-j-i);++k) {
							pos= ((Real) i*posTetra[(*itt)[0]]+
								(Real)j*posTetra[(*itt)[1]]+(Real)k*posTetra[(*itt)[2]]+
								(Real) (degreeTetrahedron-i-j-k)*posTetra[(*itt)[3]])/degreeTetrahedron;
							posTetra.push_back(pos);
							bezierTetrahedronWeight.push_back((Real)1.0f);
						}
					}
				}

			}
		}
		}
	// now translate and scale the mesh
		for (i=0;i<posTrian.size();++i) {
			posTrian[i]*=radius;
			posTrian[i]+=origin;
		}
		for (i=0;i<posTetra.size();++i) {
			posTetra[i]*=radius;
			posTetra[i]+=origin;
		}

	}



} // namespace engine

} // namespace component

} // namespace sofa

#endif
