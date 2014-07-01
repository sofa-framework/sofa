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

#include "Algo/MC/windowing.h"
#include "Topology/generic/dartmarker.h"
#include <vector>

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MC
{
template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
MarchingCube<DataType, Windowing, PFP>::MarchingCube(const char* _cName)
{
	m_Image = new Image<DataType>();

	m_Image->loadInr(_cName); // voxel sizes initialized with (1.0,1.0,1.0)
	m_Buffer = NULL;
	m_map = NULL;

	m_fOrigin = VEC3(0.0,0.0,0.0);
	m_fScal = VEC3(1.0,1.0,1.0);

	#ifdef MC_WIDTH_EDGE_Z_EMBEDED
		m_currentZSlice = 0;
		m_zslice = NULL;
	#endif
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
MarchingCube<DataType, Windowing, PFP>::MarchingCube(Image<DataType>* img, Windowing<DataType> wind, bool boundRemoved):
	m_Image(img),
	m_windowFunc(wind),
	m_Buffer(NULL),
	m_map(NULL),
	m_fOrigin(VEC3(0.0,0.0,0.0)),
	m_fScal(VEC3(1.0,1.0,1.0)),
	m_brem(boundRemoved)
{
	#ifdef MC_WIDTH_EDGE_Z_EMBEDED
		m_currentZSlice = 0;
		m_zslice = NULL;
	#endif
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
MarchingCube<DataType, Windowing, PFP>::MarchingCube(Image<DataType>* img, L_MAP* map, VertexAttribute<VEC3, L_MAP>& position, Windowing<DataType> wind, bool boundRemoved):
	m_Image(img),
	m_windowFunc(wind),
	m_Buffer(NULL),
	m_map(map),
	m_positions(position),
	m_fOrigin(VEC3(0.0,0.0,0.0)),
	m_fScal(VEC3(1.0,1.0,1.0)),
	m_brem(boundRemoved)
{
	#ifdef MC_WIDTH_EDGE_Z_EMBEDED
		m_currentZSlice = 0;
		m_zslice = NULL;
	#endif
}


template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
MarchingCube<DataType, Windowing, PFP>::~MarchingCube()
{
//	if (m_Image != NULL)
//	{
//		delete m_Image;
//	}
//
	if (m_Buffer != NULL)
	{
		delete m_Buffer;
		m_Buffer = NULL;
	}
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::deleteMesh()
{
	if (m_map != NULL)
	{
		delete m_map;
		m_map = NULL;
	}
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
Dart  MarchingCube<DataType, Windowing, PFP>::createTriEmb(unsigned int e1, unsigned int e2, unsigned int e3)
{
	L_DART d = m_map->newFace(3,false);

	unsigned int vemb = e1;

//	auto fsetemb = [&] (Dart d) { m_map->template setDartEmbedding<VERTEX>(d, vemb); };

	m_map->template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, [&] (Dart dd) { m_map->template setDartEmbedding<VERTEX>(dd, vemb); });
	d = m_map->phi1(d);
	vemb = e2;
	m_map->template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, [&] (Dart dd) { m_map->template setDartEmbedding<VERTEX>(dd, vemb); });
	d = m_map->phi1(d);
	vemb = e3;
	m_map->template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, [&] (Dart dd) { m_map->template setDartEmbedding<VERTEX>(dd, vemb); });
	d = m_map->phi1(d);

	return d;
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::simpleMeshing()
{
	// create the mesh if needed
	if (m_map == NULL)
	{
		m_map = new L_MAP();
	}

	// create the buffer
	if (m_Buffer != NULL)
	{
		delete m_Buffer;
	}

	m_Buffer = new Buffer<L_DART>(m_Image->getWidthX(),m_Image->getWidthY());

	// compute value to transform points directly to final system coordinate

	m_fOrigin   =  VEC3((float)(m_Image->getOrigin()[0]),(float)(m_Image->getOrigin()[1]),(float)(m_Image->getOrigin()[2]));

	m_fScal[0] = m_Image->getVoxSizeX();
	m_fScal[1] = m_Image->getVoxSizeY();
	m_fScal[2] = m_Image->getVoxSizeZ();

	// get access to data (pointer + size)

	DataType* ucData = m_Image->getData();

//	DataType* ucDa = ucData;

	int lTx = m_Image->getWidthX();
	int lTy = m_Image->getWidthY();
	int lTz = m_Image->getWidthZ();


	int lTxm = lTx - 1 ;
	int lTym = lTy - 1;
	int lTzm = lTz - 1;

	int lZ,lY,lX;

	lX = 0 ;
	lY = 0 ;
	lZ = 0 ;
	ucData = m_Image->getVoxelPtr(lX,lY,lZ);

	createFaces_1(ucData++,lX++,lY,lZ,1);  // TAG

	while (lX < lTxm)
	{
		createFaces_2(ucData++,lX++,lY,lZ,1);   // TAG
	}
	lY++;

	while (lY < lTym)   // 2nd and others rows  lY = 1..
	{
		lX = 0;
		ucData = m_Image->getVoxelPtr(lX,lY,lZ);

		createFaces_3(ucData++,lX++,lY,lZ,1); // TAG
		while (lX < lTxm)
		{
			createFaces_4(ucData++,lX++,lY,lZ,1);  // TAG
		}
		lY++;
	}
	lZ++;
	#ifdef MC_WIDTH_EDGE_Z_EMBEDED
		m_currentZSlice++;
	#endif

	m_Buffer->nextSlice();

// middles slices

	while (lZ < lTzm)
	{
		lY = 0;
		lX = 0;

		ucData = m_Image->getVoxelPtr(lX,lY,lZ);

		createFaces_5(ucData++,lX++,lY,lZ,4);  // TAG
		while (lX < lTxm)
		{
			createFaces_6(ucData++,lX++,lY,lZ,4);   // TAG
		}
		lY++;

		while (lY<lTym)   // 2nd and others rows  lY = 1..
		{
			lX = 0;
			ucData = m_Image->getVoxelPtr(lX,lY,lZ);

			createFaces_7(ucData++,lX++,lY,lZ,16); // TAG
			while (lX < lTxm-1)
			{
				createFaces_8(ucData++,lX++,lY,lZ,0);
			}
			createFaces_8(ucData++,lX,lY,lZ,32);   //TAG
			lY++;
		}

		lZ++;
		#ifdef MC_WIDTH_EDGE_Z_EMBEDED
			m_currentZSlice++;
		#endif
		m_Buffer->nextSlice();
	}

	CGoGNout << "Taille 2-carte:"<<m_map->getNbDarts()<<" brins"<<CGoGNendl;
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
unsigned char MarchingCube<DataType, Windowing, PFP>::computeIndex(const DataType* const _ucData) const
{
	unsigned char ucCubeIndex = 0;
	const DataType* ucDataLocal = _ucData;

	int lTx = m_Image->getWidthX();
	int lTxy = m_Image->getWidthXY();

	if ( m_windowFunc.inside(*ucDataLocal) )
		ucCubeIndex = 1; // point 0
	ucDataLocal ++;
	if ( m_windowFunc.inside(*ucDataLocal) )
		ucCubeIndex += 2; // point 1
	ucDataLocal += lTx;
	if ( m_windowFunc.inside(*ucDataLocal) )
		ucCubeIndex += 4; // point 2
	ucDataLocal --;
	if ( m_windowFunc.inside(*ucDataLocal) )
		ucCubeIndex += 8; // point 3
	ucDataLocal += lTxy;
	if ( m_windowFunc.inside(*ucDataLocal) )
		ucCubeIndex += 128; // point 7
	ucDataLocal -= lTx;
	if ( m_windowFunc.inside(*ucDataLocal) )
		ucCubeIndex += 16; // point 4
	ucDataLocal ++;
	if ( m_windowFunc.inside(*ucDataLocal) )
		ucCubeIndex += 32; // point 5
	ucDataLocal += lTx;
	if ( m_windowFunc.inside(*ucDataLocal) )
		ucCubeIndex += 64; // point 6

	return ucCubeIndex;
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
typename PFP::VEC3 MarchingCube<DataType, Windowing, PFP>::recalPoint(const VEC3& _P, const VEC3& _dec ) const
{
	VEC3 point = _P + _dec ;
//	point[0] = point[0] * m_fScal[0];
//	point[1] = point[1] * m_fScal[1];
//	point[2] = point[2] * m_fScal[2];
//
//	point += m_fOrigin;
	return 	point;
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createPointEdge0(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ,  unsigned int * const lVertTable, const VEC3& vPos)
{
	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 1)
	{
		float interp = m_windowFunc.interpole( m_Image->getVoxel(_lX,_lY,_lZ), m_Image->getVoxel(_lX+1,_lY,_lZ) );

//		VEC3 newPoint = recalPoint(vPos,VEC3(interp, 0., 0.));
//		lVertTable[0] = L_EMB::create(newPoint);
		lVertTable[0] = m_map->template newCell<VERTEX>();
		m_positions[lVertTable[0]] = recalPoint(vPos,VEC3(interp, 0., 0.));
		m_Buffer->setPointEdge0(_lX, _lY,lVertTable[0]);
	}
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createPointEdge1(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ,  unsigned int * const lVertTable, const VEC3& vPos)
{
	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 2)
	{
		float interp = m_windowFunc.interpole( m_Image->getVoxel(_lX,_lY,_lZ), m_Image->getVoxel(_lX,_lY+1,_lZ) );

//		VEC3 newPoint = recalPoint(vPos,VEC3(1.,interp, 0.));
//		lVertTable[1] = L_EMB::create(newPoint);
		lVertTable[1] = m_map->template newCell<VERTEX>();
		m_positions[lVertTable[1]] = recalPoint(vPos,VEC3(1.,interp, 0.));
		m_Buffer->setPointEdge1(_lX, _lY,lVertTable[1]);
	}
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createPointEdge2(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ,  unsigned int * const lVertTable, const VEC3& vPos)
{
	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 4)
	{
		float interp = m_windowFunc.interpole( m_Image->getVoxel(_lX-1,_lY,_lZ), m_Image->getVoxel(_lX,_lY,_lZ) );

//		VEC3 newPoint = recalPoint(vPos,VEC3(interp, 1., 0.));
//		lVertTable[2] = L_EMB::create(newPoint);
		lVertTable[2] = m_map->template newCell<VERTEX>();
		m_positions[lVertTable[2]] = recalPoint(vPos,VEC3(interp, 1., 0.));
		m_Buffer->setPointEdge2(_lX, _lY,lVertTable[2]);
	}
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createPointEdge3(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ, unsigned int * const lVertTable, const VEC3& vPos)
{
	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 8)
	{
		float interp = m_windowFunc.interpole( m_Image->getVoxel(_lX,_lY-1,_lZ), m_Image->getVoxel(_lX,_lY,_lZ) );

//		VEC3 newPoint = recalPoint(vPos,VEC3(0., interp, 0.));
//		lVertTable[3] = L_EMB::create(newPoint);
		lVertTable[3] = m_map->template newCell<VERTEX>();
		m_positions[lVertTable[3]] = recalPoint(vPos,VEC3(0., interp, 0.));
		m_Buffer->setPointEdge3(_lX, _lY,lVertTable[3]);
	}
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createPointEdge4(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ, unsigned int * const lVertTable, const VEC3& vPos)
{
	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 16)
	{
		float interp = m_windowFunc.interpole( m_Image->getVoxel(_lX,_lY,_lZ), m_Image->getVoxel(_lX+1,_lY,_lZ) );

//		VEC3 newPoint = recalPoint(vPos,VEC3(interp, 0., 1.));
//		lVertTable[4] = L_EMB::create(newPoint);
		lVertTable[4] = m_map->template newCell<VERTEX>();
		m_positions[lVertTable[4]] = recalPoint(vPos,VEC3(interp, 0., 1.));
		m_Buffer->setPointEdge4(_lX, _lY,lVertTable[4]);
	}
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createPointEdge5(const unsigned char _ucCubeIndex,  const int _lX, const int _lY, const int _lZ, unsigned int * const lVertTable, const VEC3& vPos)
{
	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 32)
 	{
		float interp = m_windowFunc.interpole( m_Image->getVoxel(_lX,_lY,_lZ), m_Image->getVoxel(_lX,_lY+1,_lZ) );

//		VEC3 newPoint = recalPoint(vPos,VEC3(1., interp, 1.));
//		lVertTable[5] = L_EMB::create(newPoint);
		lVertTable[5] = m_map->template newCell<VERTEX>();
		m_positions[lVertTable[5]] = recalPoint(vPos,VEC3(1., interp, 1.));
		m_Buffer->setPointEdge5(_lX, _lY,lVertTable[5]);
	}
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createPointEdge6(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ, unsigned int * const lVertTable, const VEC3& vPos)
{
	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 64)
	{
		float interp = m_windowFunc.interpole( m_Image->getVoxel(_lX-1,_lY,_lZ), m_Image->getVoxel(_lX,_lY,_lZ) );

//		VEC3 newPoint = recalPoint(vPos,VEC3(interp, 1., 1.));
//		lVertTable[6] = L_EMB::create(newPoint);
		lVertTable[6] = m_map->template newCell<VERTEX>();
		m_positions[lVertTable[6]] = recalPoint(vPos,VEC3(interp, 1., 1.));
		m_Buffer->setPointEdge6(_lX, _lY,lVertTable[6]);
	}
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createPointEdge7(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ, unsigned int * const lVertTable, const VEC3& vPos)
{
	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 128)
	{
		float interp = m_windowFunc.interpole( m_Image->getVoxel(_lX,_lY-1,_lZ), m_Image->getVoxel(_lX,_lY,_lZ) );

//		VEC3 newPoint = recalPoint(vPos,VEC3(0., interp, 1.));
//		lVertTable[7] = L_EMB::create(newPoint);
		lVertTable[7] = m_map->template newCell<VERTEX>();
		m_positions[lVertTable[7]] = recalPoint(vPos,VEC3(0., interp, 1.));
		m_Buffer->setPointEdge7(_lX, _lY,lVertTable[7]);
	}
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createPointEdge8(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ, unsigned int * const lVertTable, const VEC3& vPos)
{
	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 256)
	{
		float interp = m_windowFunc.interpole( m_Image->getVoxel(_lX,_lY,_lZ), m_Image->getVoxel(_lX,_lY,_lZ+1) );

//		VEC3 newPoint = recalPoint(vPos,VEC3(0., 0., interp));
//		lVertTable[8] = L_EMB::create(newPoint);
		lVertTable[8] = m_map->template newCell<VERTEX>();
		m_positions[lVertTable[8]] = recalPoint(vPos,VEC3(0., 0., interp));
		m_Buffer->setPointEdge8(_lX, _lY,lVertTable[8]);
	}
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createPointEdge9(const unsigned char _ucCubeIndex,  const int _lX, const int _lY, const int _lZ, unsigned int * const lVertTable, const VEC3& vPos)
{
	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 512)
	{
		float interp = m_windowFunc.interpole( m_Image->getVoxel(_lX,_lY,_lZ), m_Image->getVoxel(_lX,_lY,_lZ+1) );

//		VEC3 newPoint = recalPoint(vPos,VEC3(1., 0., interp));
//		lVertTable[9] = L_EMB::create(newPoint);
		lVertTable[9] = m_map->template newCell<VERTEX>();
		m_positions[lVertTable[9]] = recalPoint(vPos,VEC3(1., 0., interp));
		m_Buffer->setPointEdge9(_lX, _lY,lVertTable[9]);
	}
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createPointEdge10(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ, unsigned int * const lVertTable, const VEC3& vPos)
{
	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 1024)
	{
		float interp = m_windowFunc.interpole( m_Image->getVoxel(_lX,_lY,_lZ), m_Image->getVoxel(_lX,_lY,_lZ+1) );

//		VEC3 newPoint = recalPoint(vPos,VEC3(1., 1., interp));
//		lVertTable[10] = L_EMB::create(newPoint);
		lVertTable[10] = m_map->template newCell<VERTEX>();
		m_positions[lVertTable[10]] = recalPoint(vPos,VEC3(1., 1., interp));
		m_Buffer->setPointEdge10(_lX, _lY,lVertTable[10]);
	}
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createPointEdge11(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ, unsigned int * const lVertTable, const VEC3& vPos)
{
	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 2048)
	{
		float interp = m_windowFunc.interpole( m_Image->getVoxel(_lX,_lY,_lZ), m_Image->getVoxel(_lX,_lY,_lZ+1) );

//		VEC3 newPoint = recalPoint(vPos,VEC3(0., 1., interp));
//		lVertTable[11] = L_EMB::create(newPoint);
		lVertTable[11] = m_map->template newCell<VERTEX>();
		m_positions[lVertTable[11]] = recalPoint(vPos,VEC3(0., 1., interp));
		m_Buffer->setPointEdge11(_lX, _lY,lVertTable[11]);
	}
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createFaces_1(DataType *vox, const int _lX,const int _lY,const int _lZ, unsigned char tag)
{
	unsigned char ucCubeIndex = computeIndex(vox);
	if ((ucCubeIndex == 0) || (ucCubeIndex == 255))
		return;

	unsigned int  lVertTable[12];

//	VEC3 vPos(float(_lX) , float(_lY) , float(_lZ) );
//	VEC3 vPos(float(_lX) + 0.5f, float(_lY) + 0.5f, (float)_lZ + 0.5f);
	VEC3 vPos(_lX, _lY, _lZ);

// create the new  vertices

	int lX = _lX;
	int lY = _lY;
	int lZ = _lZ;

//	float curv = m_Curvature.computeCurvatureSimple(vox);
	float curv = 0.0f;
	createPointEdge0( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	createPointEdge8( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lX++;
	createPointEdge1( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	createPointEdge9( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lY++;
	createPointEdge2( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	createPointEdge10( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lX--;
	createPointEdge3( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	createPointEdge11( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lZ++;
	createPointEdge7( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lY--;
	createPointEdge4( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lX++;
	createPointEdge5( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lY++;
	createPointEdge6( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);

// get the shared vertices corresponding to the shared edges :
	//all edges
	    lVertTable[ 0]  = m_Buffer->getPointEdge0(_lX, _lY);
	    lVertTable[ 1]  = m_Buffer->getPointEdge1(_lX, _lY);
	    lVertTable[ 2]  = m_Buffer->getPointEdge2(_lX, _lY);
	    lVertTable[ 3]  = m_Buffer->getPointEdge3(_lX, _lY);
	    lVertTable[ 4]  = m_Buffer->getPointEdge4(_lX, _lY);
	    lVertTable[ 5]  = m_Buffer->getPointEdge5(_lX, _lY);
	    lVertTable[ 6]  = m_Buffer->getPointEdge6(_lX, _lY);
	    lVertTable[ 7]  = m_Buffer->getPointEdge7(_lX, _lY);
	    lVertTable[ 8]  = m_Buffer->getPointEdge8(_lX, _lY);
	    lVertTable[ 9]  = m_Buffer->getPointEdge9(_lX, _lY);
	    lVertTable[10]  = m_Buffer->getPointEdge10(_lX, _lY);
	    lVertTable[11]  = m_Buffer->getPointEdge11(_lX, _lY);

// create the new faces, the mask (6 bits) of faces to treat is 111111
	createLocalFaces(ucCubeIndex, _lX, _lY, _lZ,  lVertTable, 0x3F, curv, tag);
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createFaces_2(DataType *vox, const int _lX, const int _lY, const int _lZ, unsigned char tag)
{
	unsigned char ucCubeIndex = computeIndex(vox);
	if ((ucCubeIndex == 0) || (ucCubeIndex == 255))
		return;

	unsigned int  lVertTable[12];
	VEC3 vPos(_lX, _lY, _lZ);

// create the new  vertices
	int lX = _lX;
	int lY = _lY;
	int lZ = _lZ;

//	float curv = m_Curvature.computeCurvatureSimple(vox);
	float curv=0.0f;

	createPointEdge0( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos );
	lX++;
	createPointEdge1( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos );
	createPointEdge9( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos );
	lY++;
	createPointEdge2( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	createPointEdge10( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lX--;
	lZ++;
	lY--;
	createPointEdge4( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lX++;
	createPointEdge5( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos );
	lY++;
	createPointEdge6( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);

// get the shared vertices corresponding to the shared edges :
	//edge 3
	    lVertTable[3]  = m_Buffer->getPointEdge3(_lX, _lY);

	// edge 7
	    lVertTable[7]  = m_Buffer->getPointEdge7(_lX, _lY);

	// edge 8
	    lVertTable[8]  = m_Buffer->getPointEdge8(_lX, _lY);

	//edge 11
	    lVertTable[11] = m_Buffer->getPointEdge11(_lX, _lY);

// create the new faces, the mask (6 bits) of faces to treat is 111110
	createLocalFaces(ucCubeIndex, _lX, _lY, _lZ,  lVertTable, 0x3E, curv, tag);
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createFaces_3(DataType *vox, const int _lX, const int _lY, const int _lZ, unsigned char tag)
{
	unsigned char ucCubeIndex = computeIndex(vox);
	if ((ucCubeIndex == 0) || (ucCubeIndex == 255))
		return;

	unsigned int  lVertTable[12];
	VEC3 vPos(_lX, _lY, _lZ);

// create the new  vertices
	int lX = _lX;
	int lY = _lY;
	int lZ = _lZ;

//	float curv = m_Curvature.computeCurvatureSimple(vox);
	float curv=0.0f;

	lX++;
	createPointEdge1( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lY++;
	createPointEdge2( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	createPointEdge10( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lX--;
	createPointEdge3( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	createPointEdge11( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lZ++;
	createPointEdge7( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lY--;
	lX++;
	createPointEdge5( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lY++;
	createPointEdge6( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);

// get the shared vertices corresponding to the shared edges :
	//edge 0
	    lVertTable[0] = m_Buffer->getPointEdge0(_lX, _lY);

	// edge 9
	    lVertTable[9] = m_Buffer->getPointEdge9(_lX, _lY);

	// edge 4
	    lVertTable[4] = m_Buffer->getPointEdge4(_lX, _lY);

	//edge 8
	    lVertTable[8] = m_Buffer->getPointEdge8(_lX, _lY);


// create the new faces, the mask (6 bits) of faces to treat is 111011
	createLocalFaces(ucCubeIndex, _lX, _lY, _lZ,  lVertTable, 0x3B, curv, tag);
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createFaces_4(DataType *vox, const int _lX, const int _lY, const int _lZ, unsigned char tag)
{
	unsigned char ucCubeIndex = computeIndex(vox);
	if ((ucCubeIndex == 0) || (ucCubeIndex == 255))
		return;

	unsigned int  lVertTable[12];
	VEC3 vPos(_lX, _lY, _lZ);

// create the new  vertices
	int lX = _lX;
	int lY = _lY;
	int lZ = _lZ;

//	float curv = m_Curvature.computeCurvatureSimple(vox);
	float curv=0.0f;

	lX++;
	createPointEdge1( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lY++;
	createPointEdge2( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	createPointEdge10( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lZ++;
	lY--;
	createPointEdge5( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lY++;
	createPointEdge6( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);

// get the shared vertices corresponding to the shared edges :
	//edge 3
	    lVertTable[3]  = m_Buffer->getPointEdge3(_lX, _lY);

	// edge 7
	    lVertTable[7]  = m_Buffer->getPointEdge7(_lX, _lY);

	// edge 8
	    lVertTable[8]  = m_Buffer->getPointEdge8(_lX, _lY);

	//edge 11
	    lVertTable[11] = m_Buffer->getPointEdge11(_lX, _lY);

	//edge 0
	    lVertTable[0] = m_Buffer->getPointEdge0(_lX, _lY);

	// edge 9
	    lVertTable[9] = m_Buffer->getPointEdge9(_lX, _lY);

	// edge 4
	    lVertTable[4] = m_Buffer->getPointEdge4(_lX, _lY);

// create the new faces, the mask (6 bits) of faces to treat is 111010
	createLocalFaces(ucCubeIndex, _lX, _lY, _lZ,  lVertTable, 0x3A, curv, tag);
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createFaces_5(DataType *vox, const int _lX, const int _lY, const int _lZ, unsigned char tag)
{
	unsigned char ucCubeIndex = computeIndex(vox);
	if ((ucCubeIndex == 0) || (ucCubeIndex == 255))
		return;

	unsigned int  lVertTable[12];
	VEC3 vPos(_lX, _lY, _lZ);

// create the new  vertices
	int lX = _lX;
	int lY = _lY;
	int lZ = _lZ;

//	float curv = m_Curvature.computeCurvatureSimple(vox);
	float curv=0.0f;

	createPointEdge8( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lX++;
	createPointEdge9( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lY++;
	createPointEdge10( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lX--;
	createPointEdge3( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	createPointEdge11( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lZ++;
	createPointEdge7( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lY--;
	createPointEdge4( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lX++;
	createPointEdge5( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lY++;
	createPointEdge6( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);

// get the shared vertices corresponding to the shared edges :
	//edge 0
	    lVertTable[0] = m_Buffer->getPointEdge0(_lX, _lY);

	// edge 1
	    lVertTable[1] = m_Buffer->getPointEdge1(_lX, _lY);

	// edge 2
	    lVertTable[2] = m_Buffer->getPointEdge2(_lX, _lY);

	//edge 3
	    lVertTable[3] = m_Buffer->getPointEdge3(_lX, _lY);


// create the new faces, the mask (6 bits) of faces to treat is 101111
	createLocalFaces(ucCubeIndex, _lX, _lY, _lZ,  lVertTable, 0x2F, curv, tag);
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createFaces_6(DataType *vox, const int _lX, const int _lY, const int _lZ, unsigned char tag)
{
	unsigned char ucCubeIndex = computeIndex(vox);
	if ((ucCubeIndex == 0) || (ucCubeIndex == 255))
		return;

	unsigned int  lVertTable[12];
	VEC3 vPos(_lX, _lY, _lZ);

// create the new  vertices
	int lX = _lX+1;
	int lY = _lY;
	int lZ = _lZ;

	vox++;

//	float curv = m_Curvature.computeCurvatureSimple(vox);
	float curv=0.0f;
	createPointEdge9( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lY++;
	createPointEdge10( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lX--;
	lZ++;
	lY--;
	createPointEdge4( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lX++;
	createPointEdge5( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lY++;
	createPointEdge6( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);

// get the shared vertices corresponding to the shared edges :
	//edge 0
	    lVertTable[0] = m_Buffer->getPointEdge0(_lX, _lY);

	// edge 1
	    lVertTable[1] = m_Buffer->getPointEdge1(_lX, _lY);

	// edge 2
	    lVertTable[2] = m_Buffer->getPointEdge2(_lX, _lY);

	//edge 3
	    lVertTable[3] = m_Buffer->getPointEdge3(_lX, _lY);

	// edge 7
	    lVertTable[7] = m_Buffer->getPointEdge7(_lX, _lY);

	// edge 8
	    lVertTable[8] = m_Buffer->getPointEdge8(_lX, _lY);

	// edge 11
	    lVertTable[11] = m_Buffer->getPointEdge11(_lX, _lY);


// create the new faces, the mask (6 bits) of faces to treat is 101110
	createLocalFaces(ucCubeIndex, _lX, _lY, _lZ,  lVertTable, 0x2E, curv, tag);
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createFaces_7(DataType *vox, const int _lX, const int _lY, const int _lZ, unsigned char tag)
{
	unsigned char ucCubeIndex = computeIndex(vox);
	if ((ucCubeIndex == 0) || (ucCubeIndex == 255))
		return;

	unsigned int  lVertTable[12];
	VEC3 vPos(_lX, _lY, _lZ);

// create the new  vertices
	int lX = _lX+1;
	int lY = _lY+1;
	int lZ = _lZ;

	vox += m_Image->getWidthX()+1;

//	float curv = m_Curvature.computeCurvatureSimple(vox);
	float curv=0.0f;

	createPointEdge10( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lX--;
	createPointEdge11( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lZ++;
	createPointEdge7( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lY--;
	lX++;
	createPointEdge5( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	lY++;
	createPointEdge6( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);

// get the shared vertices corresponding to the shared edges :
	//edge 0
	    lVertTable[0] = m_Buffer->getPointEdge0(_lX, _lY);

	// edge 1
	    lVertTable[1] = m_Buffer->getPointEdge1(_lX, _lY);

	// edge 2
	    lVertTable[2] = m_Buffer->getPointEdge2(_lX, _lY);

	//edge 3
	    lVertTable[3] = m_Buffer->getPointEdge3(_lX, _lY);

	// edge 4
	    lVertTable[4] = m_Buffer->getPointEdge4(_lX, _lY);

	// edge 8
	    lVertTable[8] = m_Buffer->getPointEdge8(_lX, _lY);

	//edge 9
	    lVertTable[9] = m_Buffer->getPointEdge9(_lX, _lY);


// create the new faces, the mask (6 bits) of faces to treat is 101011
	createLocalFaces(ucCubeIndex, _lX, _lY, _lZ,  lVertTable, 0x2B, curv, tag);
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createFaces_8(DataType *vox, const int _lX, const int _lY, const int _lZ, unsigned char tag)
{
	unsigned char ucCubeIndex = computeIndex(vox);
	if ((ucCubeIndex == 0) || (ucCubeIndex == 255))
		return;

	unsigned int  lVertTable[12];
	VEC3 vPos(_lX, _lY, _lZ);

// create the new  vertices
	int lX = _lX+1;
	int lY = _lY+1;
	int lZ = _lZ+1;

//	float curv = m_Curvature.computeCurvatureSimple(vox);
	float curv=0.0f;

	createPointEdge6( ucCubeIndex, lX,  lY,  lZ, lVertTable, vPos);
	createPointEdge5( ucCubeIndex, lX,  _lY,  lZ, lVertTable, vPos);
	createPointEdge10( ucCubeIndex, lX,  lY,  _lZ, lVertTable, vPos);

// get the shared vertices corresponding to the shared edges :
	//edge 0
	    lVertTable[0] = m_Buffer->getPointEdge0(_lX, _lY);

	// edge 1
	    lVertTable[1] = m_Buffer->getPointEdge1(_lX, _lY);

	// edge 2
	    lVertTable[2] = m_Buffer->getPointEdge2(_lX, _lY);

	//edge 3
	    lVertTable[3] = m_Buffer->getPointEdge3(_lX, _lY);

	// edge 4
	    lVertTable[4] = m_Buffer->getPointEdge4(_lX, _lY);

	// edge 8
	    lVertTable[8] = m_Buffer->getPointEdge8(_lX, _lY);

	//edge 9
	    lVertTable[9] = m_Buffer->getPointEdge9(_lX, _lY);

	// edge 7
	    lVertTable[7] = m_Buffer->getPointEdge7(_lX, _lY);

	// edge 11
	    lVertTable[11] = m_Buffer->getPointEdge11(_lX, _lY);


// create the new faces, the mask (6 bits) of faces to treat is 101010
	createLocalFaces(ucCubeIndex, _lX, _lY, _lZ,  lVertTable, 0x2A, curv, tag);
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::setNeighbourSimple(L_DART d1, L_DART d2)
{
	if (m_map->phi2(d1) != d2)
	{
		m_map->sewFaces(d1,d2,false);
		#ifdef MC_WIDTH_EDGE_Z_EMBEDED
		if (m_zslice!=NULL)
		{
			unsigned int val = (m_currentZSlice - m_zbound)/m_sliceGroup;
			std::cout << "ZSLICE=" << val << std::endl;
			m_zslice->operator[](d1) = val;
		}
		#endif
	}
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::setNeighbour(L_DART d1, L_DART d2)
{
	if (m_map->phi2(d1) != d2)
	{
		m_map->sewFaces(d1,d2,false);
		#ifdef MC_WIDTH_EDGE_Z_EMBEDED
		if (m_zslice!=NULL)
		{
			unsigned int val = (m_currentZSlice - m_zbound)/m_sliceGroup;
			std::cout << "ZSLICE=" << val << std::endl;
			m_zslice->operator[](d1) = val;
		}
		#endif
	}
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::createLocalFaces(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int /*_lZ*/,  unsigned int  const *_lVertTable, const unsigned short _usMask, float /*curv*/, unsigned char /*tag*/)
{
// TODO parametre _LZ not used => a supprimer ?
// TODO parametre curv not used => a supprimer ?
// TODO parametre tag not used => a supprimer ?
	
	#define EVENMASK	0x55
	#define ODDMASK	0xAA

// initialize number of created face to 0
	int lNumFaces = 0;

// get the address of buffer table
	L_DART *lFacesTab = m_Buffer->getFacesCubeTableAdr(_lX,_lY);

// To speed access to data "precompute indirection"
	const char* cTriangle = accelMCTable::m_TriTable[_ucCubeIndex];
	const char* cNeighbour = accelMCTable::m_NeighTable[_ucCubeIndex];

//	L_DART dartTriangles[5];

// PASS 1: create faces
 	for (int i=0; cTriangle[i]!=-1;/* incrementation inside */)
	{
	// get index of the 3 points of trianglesfrzr
		int lIndexP1 =  cTriangle[i++];
		int lIndexP2 =  cTriangle[i++];
		int lIndexP3 =  cTriangle[i++];

/*		L_DART edge = m_map->createEmbOriTriangle(_lVertTable[lIndexP1],  //first point/edge
							_lVertTable[lIndexP2], //second point/edge
							_lVertTable[lIndexP3]);// third point/edge
*/
		L_DART edge = createTriEmb(_lVertTable[lIndexP1],  //first point/edge
							_lVertTable[lIndexP2], //second point/edge
							_lVertTable[lIndexP3]);// third point/edge

	// add the face to buffer
		lFacesTab[lNumFaces] = edge;
		lNumFaces++;
	}

// PASS 2: update neighbours
	for (int j=0; j<lNumFaces;++j)
	{
		int i = 3*j;

		int lIndexP1 =  cTriangle[i];
		int lIndexP2 =  cTriangle[i+1];
		int lIndexP3 =  cTriangle[i+2];

		L_DART edge = lFacesTab[j];

		int lLocalNeighbour = cNeighbour[i];
		L_DART dn = lFacesTab[ lLocalNeighbour/3 ];
		if (lLocalNeighbour>0)
			lLocalNeighbour = lLocalNeighbour%3;
		switch(lLocalNeighbour)
		{
			case 0:
				setNeighbourSimple(edge,dn);
				break;
			case 1:
				dn =m_map->phi1(dn);
				setNeighbourSimple(edge,dn);
				break;
			case 2:
				dn = m_map->phi_1(dn); // triangle donc pas besoin de faire le tour par 2xphi1
				setNeighbourSimple(edge,dn);
				break;
			default: // -1
				unsigned char ucCodeEdgeFace = accelMCTable::m_EdgeCode[lIndexP1] & accelMCTable::m_EdgeCode[lIndexP2];
				if (ucCodeEdgeFace & EVENMASK)	// the edge has a neighbour
				{
					L_DART neighbour = m_Buffer->getExternalNeighbour(lIndexP1 , _lX, _lY);	// get neighbour from buffer
					setNeighbour(edge,neighbour);
				}
				break;
		}

		// second edge
		edge =m_map->phi1(edge);
		lLocalNeighbour = cNeighbour[i+1];
		dn = lFacesTab[lLocalNeighbour/3];
		if (lLocalNeighbour>0)
			lLocalNeighbour = lLocalNeighbour%3;
		switch(lLocalNeighbour)
		{
			case 0:
				setNeighbourSimple(edge,dn);
				break;
			case 1:
				dn =m_map->phi1(dn);
				setNeighbourSimple(edge,dn);
				break;
			case 2:
				dn = m_map->phi_1(dn); // triangle donc pas besoin de faire le tour par 2xphi1
				setNeighbourSimple(edge,dn);
				break;
			default: // -1
				unsigned char ucCodeEdgeFace = accelMCTable::m_EdgeCode[lIndexP2] & accelMCTable::m_EdgeCode[lIndexP3];
				if (ucCodeEdgeFace & EVENMASK)	// the edge has a neighbour
				{
					L_DART neighbour = m_Buffer->getExternalNeighbour(lIndexP2 , _lX, _lY);	// get neighbour from buffer
					setNeighbour(edge,neighbour);
				}
				break;
		}

		// third edge
		edge =m_map->phi1(edge);
		lLocalNeighbour = cNeighbour[i+2];
		dn = lFacesTab[lLocalNeighbour/3];
		if (lLocalNeighbour>0)
			lLocalNeighbour = lLocalNeighbour%3;
		switch(lLocalNeighbour)
		{
			case 0:
				setNeighbourSimple(edge,dn);
				break;
			case 1:
				dn =m_map->phi1(dn);
				setNeighbourSimple(edge,dn);
				break;
			case 2:
				dn = m_map->phi_1(dn); // triangle donc pas besoin de faire le tour par 2xphi1
				setNeighbourSimple(edge,dn);
				break;
			default: // -1
				unsigned char ucCodeEdgeFace = accelMCTable::m_EdgeCode[lIndexP3] & accelMCTable::m_EdgeCode[lIndexP1];
				if (ucCodeEdgeFace & EVENMASK)	// the edge has a neighbour
				{
					L_DART neighbour = m_Buffer->getExternalNeighbour(lIndexP3 , _lX, _lY);	// get neighbour from buffer
					setNeighbour(edge,neighbour);
				}
				break;
		}
	}

// PASS 2: set the neighbours in buffer

 	for (int i=0; cTriangle[i]!=-1;/* incrementation inside */)
	{
		L_DART edge = lFacesTab[i/3];

		int lIndexP1 =  cTriangle[i];
		int lIndexP2 =  cTriangle[i+1];
		int lIndexP3 =  cTriangle[i+2];

		if (cNeighbour[i] < 0) // update neighbourhood internal to cube
		{
			// compute the code of the edge (define which face of cube it belongs)
			unsigned char ucCodeEdgeFace = accelMCTable::m_EdgeCode[lIndexP1] & accelMCTable::m_EdgeCode[lIndexP2];
			if (ucCodeEdgeFace & _usMask & ODDMASK)	// the neighbour of the edge not yet created
			{
				m_Buffer->setExternalNeighbour(lIndexP2 , _lX, _lY,  edge);	// set the edge in edge of P2 in the buffer
			}
		}

		edge =m_map->phi1(edge); // next edge
		i++;

		if (cNeighbour[i] < 0)  // update neighbourhood internal to cube
		{
			// compute the code of the edge (define which face of cube it belongs)
			unsigned char ucCodeEdgeFace = accelMCTable::m_EdgeCode[lIndexP2] & accelMCTable::m_EdgeCode[lIndexP3];

			if (ucCodeEdgeFace & _usMask & ODDMASK)	// the neighbour of the edge not yet created
			{
				m_Buffer->setExternalNeighbour(lIndexP3 , _lX, _lY,  edge);	// set the edge in edge of P2 in the buffer
			}
		}

		edge =m_map->phi1(edge); // next edge
		i++;

   		if (cNeighbour[i] < 0)  // update neighbourhood internal to cube
		{
			// compute the code of the edge (define which face of cube it belongs)
			unsigned char ucCodeEdgeFace = accelMCTable::m_EdgeCode[lIndexP3] & accelMCTable::m_EdgeCode[lIndexP1];
			if (ucCodeEdgeFace & _usMask & ODDMASK)	// the neighbour of the edge not yet created
			{
				m_Buffer->setExternalNeighbour(lIndexP1 , _lX, _lY,  edge);	// set the edge in edge of P2 in the buffer
 			}
		}
		i++;
	}

// finish buffer table of faces with -1
// PAS FORCEMENT UTILE A VERIFIER
// 	for(int i=lNumFaces; i <5; i++)
// 	{
// 		lFacesTab[i] = m_map->end();
// 	}

	#undef EVENMASK
	#undef ODDMASK
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::removeFacesOfBoundary(VertexAttribute<unsigned char, L_MAP>& boundVertices, unsigned int frameWidth)
{
	float xmin = frameWidth;
	float xmax = m_Image->getWidthX() - frameWidth -1;
	float ymin = frameWidth;
	float ymax = m_Image->getWidthY() - frameWidth -1;
	float zmin = frameWidth;
	float zmax = m_Image->getWidthZ() - frameWidth -1;

	// traverse position and create bound attrib
	for(unsigned int it = m_positions.begin(); it != m_positions.end(); m_positions.next(it))
	{
		bool bound = (m_positions[it][0]<=xmin) || (m_positions[it][0]>=xmax) || \
					 (m_positions[it][1]<=ymin) || (m_positions[it][1]>=ymax) || \
					 (m_positions[it][2]<=zmin) || (m_positions[it][2]>=zmax);

		if (bound)
		{
			boundVertices[it] = 1;
		}
		else
			boundVertices[it] = 0;
	}

//	 traverse face and check if all vertices are bound
	DartMarker<L_MAP> mf(*m_map);
	for (Dart d = m_map->begin(); d != m_map->end();)	// next done inside loop because of deleteFace
	{
		if (!mf.isMarked(d) && !m_map->isBoundaryMarked2(d))
		{
			Dart dd = d;
			Dart e = m_map->phi1(d);
			Dart f = m_map->phi1(e);
			m_map->next(d);
			while ((d==e) || (d==f))
			{
				m_map->next(d);
			}
			if ((boundVertices[dd]!=0) && (boundVertices[e]!=0) && (boundVertices[f]!=0))
				m_map->deleteFace(dd,false);
			else
				mf.markOrbit<FACE>(dd);
		}
		else m_map->next(d);
	}
	m_map->closeMap();

////	 VERSION USING DELETE FACE WITH BOUNDARY
//	DartMarker mf(*m_map);
//	std::vector<Dart> vecF;
//	vecF.reserve(8192);
//	for (Dart d = m_map->begin(); d != m_map->end();m_map->next(d))	// next done inside loop because of deleteFace
//	{
//		if ((!mf.isMarked(d)) && (!m_map->isBoundaryMarked2(d)) )
//		{
//			Dart dd = d;
//			Dart e = m_map->phi1(d);
//			Dart f = m_map->phi1(e);
//			if ((boundVertices[dd]!=0) && (boundVertices[e]!=0) && (boundVertices[f]!=0))
//				vecF.push_back(d);
//			mf.markOrbit(FACE,dd);
//		}
//	}
//	for (std::vector<Dart>::iterator it = vecF.begin(); it != vecF.end(); ++it)
//		m_map->deleteFace(*it);
}

template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::recalPoints(const Geom::Vec3f& origin)
{
	for(unsigned int i=m_positions.begin(); i != m_positions.end(); m_positions.next(i))
	{
		VEC3& P = m_positions[i];
		P -= m_fOrigin;
		P[0] = (P[0]+0.5f) * m_fScal[0];
		P[1] = (P[1]+0.5f) * m_fScal[1];
		P[2] = (P[2]+0.5f) * m_fScal[2];
		P+=origin;
	}
}


#ifdef MC_WIDTH_EDGE_Z_EMBEDED
template< typename  DataType, template < typename D2 > class Windowing, typename PFP >
void MarchingCube<DataType, Windowing, PFP>::setZSliceAttrib(EdgeAttribute<unsigned long long>* zsatt, unsigned int zbound, unsigned int nbZone)
{
	m_zslice = zsatt;
	m_zbound = zbound;
	m_sliceGroup = m_Image->getWidthZ()/nbZone;
}
#endif


} // namespace MC

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
