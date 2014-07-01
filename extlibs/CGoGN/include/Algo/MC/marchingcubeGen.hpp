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

#include "windowing.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MC
{

// template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
// MarchingCubeGen<DataType, ImgT, Windowing, PFP>::MarchingCubeGen(const char* _cName)
// {
// 	m_Image = new ImgT<DataType>();
//
// //	m_Image->loadInr(_cName); // voxel sizes initialized with (1.0,1.0,1.0)
// 	m_Buffer = NULL;
// 	m_map = NULL;
//
// 	m_fOrigin = typename PFP::VEC3(0.0,0.0,0.0);
// 	m_fScal = typename PFP::VEC3(1.0,1.0,1.0);
// }


template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
MarchingCubeGen<DataType, ImgT, Windowing, PFP>::MarchingCubeGen(ImgT* img, Windowing<DataType> wind, bool boundRemoved):
	m_Image(img),
	m_windowFunc(wind),
	m_Buffer(NULL),
	m_map(NULL),
	m_fOrigin(typename PFP::VEC3(0.0,0.0,0.0)),
	m_fScal(typename PFP::VEC3(1.0,1.0,1.0)),
	m_brem(boundRemoved)
{
}

template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
MarchingCubeGen<DataType, ImgT, Windowing, PFP>::MarchingCubeGen(ImgT* img, L_MAP* map, unsigned int idPos, Windowing<DataType> wind, bool boundRemoved):
	m_Image(img),
	m_windowFunc(wind),
	m_Buffer(NULL),
	m_map(map),
	m_positions(idPos,*map),
	m_fOrigin(typename PFP::VEC3(0.0,0.0,0.0)),
	m_fScal(typename PFP::VEC3(1.0,1.0,1.0)),
	m_brem(boundRemoved)
{
}




template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
MarchingCubeGen<DataType, ImgT, Windowing, PFP>::~MarchingCubeGen()
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

template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::deleteMesh()
{
	if (m_map != NULL)
	{
		delete m_map;
		m_map = NULL;
	}
}


template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, typename PFP >
Dart  MarchingCube<DataType, ImgT, Windowing, PFP>::createTriEmb(unsigned int e1, unsigned int e2, unsigned int e3)
{
	L_DART d = m_map->newFace(3,false);
		
	FunctorSetEmb<typename PFP::MAP, VERTEX> fsetemb(*m_map, e1);
	m_map->template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
	d = m_map->phi1(d);
	fsetemb.changeEmb(e2);
	m_map->template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
	d = m_map->phi1(d);
	fsetemb.changeEmb(e3);
	m_map->template foreach_dart_of_orbit<PFP::MAP::VERTEX_OF_PARENT>(d, fsetemb);
	d = m_map->phi1(d);

	return d;
}

template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::simpleMeshing()
{


	// create the mesh if needed
	if (m_map==NULL)
	{
		m_map = new L_MAP();
	}

	// create the buffer
	if (m_Buffer != NULL)
	{
		delete m_Buffer;
	}

	m_Buffer = new BufferGen<L_DART,DataType>(m_Image->getWidthX(),m_Image->getWidthY());

	// compute value to transform points directly to final system coordinate

/*	m_fOrigin   =  typename PFP::VEC3((float)(m_Image->getOrigin()[0]),(float)(m_Image->getOrigin()[1]),(float)(m_Image->getOrigin()[2]));*/

	m_fScal[0] = m_Image->getVoxSizeX();
	m_fScal[1] = m_Image->getVoxSizeY();
	m_fScal[2] = m_Image->getVoxSizeZ();



	// get access to data (pointer + size)

// 	DataType* ucData = m_Image->getData();
//
// 	DataType* ucDa = ucData;

	int lTx = m_Image->getWidthX();
	int lTy = m_Image->getWidthY();
	int lTz = m_Image->getWidthZ();


/*	gmtl::Vec3i orig = m_Image->getOrigin();*/

	int lTxm = lTx - 1 ;
	int lTym = lTy - 1;
	int lTzm = lTz - 1;

	int lZ,lY,lX;

	lX = 0 ;
	lY = 0 ;
	lZ = 0 ;

	createFaces_1(lX++,lY,lZ,1);  // TAG

	while (lX < lTxm)
	{
		createFaces_2(lX++,lY,lZ,1);   // TAG
	}
	lY++;

	while (lY < lTym)   // 2nd and others rows  lY = 1..
	{
		lX = 0;

		createFaces_3(lX++,lY,lZ,1); // TAG
		while (lX < lTxm)
		{
			createFaces_4(lX++,lY,lZ,1);  // TAG
		}
		lY++;
	}
	lZ++;
	m_Buffer->nextSlice();

// middles slices

	while (lZ < lTzm)
	{
		lY = 0;
		lX = 0;

		createFaces_5(lX++,lY,lZ,4);  // TAG
		while (lX < lTxm)
		{
			createFaces_6(lX++,lY,lZ,4);   // TAG
		}
		lY++;

		while (lY<lTym)   // 2nd and others rows  lY = 1..
		{
			lX = 0;

			createFaces_7(lX++,lY,lZ,16); // TAG
			while (lX < lTxm)
			{
				createFaces_8(lX++,lY,lZ,0);
			}
			lY++;
		}

		lZ++;
		m_Buffer->nextSlice();
	}

	CGoGNout << "Taille 2-carte:"<<m_map->size()<<" brins"<<CGoGNendl;

}



template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
unsigned char MarchingCubeGen<DataType, ImgT, Windowing, PFP>::computeIndex(int lX, int lY) const
{
	unsigned char ucCubeIndex = 0;
	int bwidth = m_Buffer->getWidth();
	int dec =  lX + lY*bwidth;

	DataType* dat = m_Buffer->getDataPtr() + dec;

	if ( m_windowFunc.inside(*dat) )
		ucCubeIndex = 1; // point 0
	dat++;
	if ( m_windowFunc.inside(*dat) )
		ucCubeIndex += 2; // point 1
	dat += bwidth;
	if ( m_windowFunc.inside(*dat) )
		ucCubeIndex += 4; // point 2
	dat--;
	if ( m_windowFunc.inside(*dat) )
		ucCubeIndex += 8; // point 3

	dat = m_Buffer->getData2Ptr() + dec;

	if ( m_windowFunc.inside(*dat) )
		ucCubeIndex += 16; // point 0
	dat++;
	if ( m_windowFunc.inside(*dat) )
		ucCubeIndex += 32; // point 1
	dat += bwidth;
	if ( m_windowFunc.inside(*dat) )
		ucCubeIndex += 64; // point 2
	dat--;
	if ( m_windowFunc.inside(*dat) )
		ucCubeIndex += 128; // point 3

	return ucCubeIndex;
}

template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
typename PFP::VEC3 MarchingCubeGen<DataType, ImgT, Windowing, PFP>::recalPoint(const typename PFP::VEC3& _P, const typename PFP::VEC3& _dec ) const
{
	typename PFP::VEC3 point = _P + _dec ;

	point -= m_fOrigin;

	point[0] = point[0] * m_fScal[0];
	point[1] = point[1] * m_fScal[1];
	point[2] = point[2] * m_fScal[2];

	return 	point;
}



template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createPointEdge0(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ,  unsigned int* const lVertTable, const typename PFP::VEC3& vPos)
{
// TODO parametre _LZ not used => a supprimer ?

	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 1)
	{
		float interp = m_windowFunc.interpole( m_Buffer->getData(_lX,_lY), m_Buffer->getData(_lX+1,_lY) );

		lVertTable[0] = m_map->newCell(VERTEX);
		m_positions[lVertTable[0]] = recalPoint(vPos,typename PFP::VEC3(interp, 0., 0.));
		m_Buffer->setPointEdge0(_lX, _lY,lVertTable[0]);


	}
}



template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createPointEdge1(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ,  unsigned int* const lVertTable, const typename PFP::VEC3& vPos)
{
// TODO parametre _LZ not used => a supprimer ?

	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 2)
	{
		float interp = m_windowFunc.interpole( m_Buffer->getData(_lX,_lY), m_Buffer->getData(_lX,_lY+1) );

		lVertTable[1] = m_map->newCell(VERTEX);
		m_positions[lVertTable[1]] = recalPoint(vPos,typename PFP::VEC3(1.,interp, 0.));
		m_Buffer->setPointEdge1(_lX, _lY,lVertTable[1]);
	}
}


template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createPointEdge2(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ,  unsigned int* const lVertTable, const typename PFP::VEC3& vPos)
{
// TODO parametre _LZ not used => a supprimer ?

	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 4)
	{
		float interp = m_windowFunc.interpole( m_Buffer->getData(_lX-1,_lY), m_Buffer->getData(_lX,_lY) );

		lVertTable[2] = m_map->newCell(VERTEX);
		m_positions[lVertTable[2]] = recalPoint(vPos,typename PFP::VEC3(interp, 1., 0.));
		m_Buffer->setPointEdge2(_lX, _lY,lVertTable[2]);
	}
}


template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createPointEdge3(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ, unsigned int* const lVertTable, const typename PFP::VEC3& vPos)
{
// TODO parametre _LZ not used => a supprimer ?
	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 8)
	{
		float interp = m_windowFunc.interpole( m_Buffer->getData(_lX,_lY-1), m_Buffer->getData(_lX,_lY-1) );

		lVertTable[3] = m_map->newCell(VERTEX);
		m_positions[lVertTable[3]] = recalPoint(vPos,typename PFP::VEC3(0., interp, 0.));
		m_Buffer->setPointEdge3(_lX, _lY,lVertTable[3]);
	}
}


template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createPointEdge4(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ, unsigned int* const lVertTable, const typename PFP::VEC3& vPos)
{
// TODO parametre _LZ not used => a supprimer ?

	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 16)
	{
		float interp = m_windowFunc.interpole( m_Buffer->getData2(_lX,_lY), m_Buffer->getData2(_lX+1,_lY) );

		lVertTable[4] = m_map->newCell(VERTEX);
		m_positions[lVertTable[4]] = recalPoint(vPos,typename PFP::VEC3(interp, 0., 1.));
		m_Buffer->setPointEdge4(_lX, _lY,lVertTable[4]);
	}
}


template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createPointEdge5(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ, unsigned int* const lVertTable, const typename PFP::VEC3& vPos)
{
// TODO parametre _LZ not used => a supprimer ?

	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 32)
 	{
		float interp = m_windowFunc.interpole( m_Buffer->getData2(_lX,_lY), m_Buffer->getData2(_lX,_lY+1) );

		lVertTable[5] = m_map->newCell(VERTEX);
		m_positions[lVertTable[5]] = recalPoint(vPos,typename PFP::VEC3(1., interp, 1.));
		m_Buffer->setPointEdge5(_lX, _lY,lVertTable[5]);
	}
}


template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createPointEdge6(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ, unsigned int* const lVertTable, const typename PFP::VEC3& vPos)
{
// TODO parametre _LZ not used => a supprimer ?

	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 64)
	{
		float interp = m_windowFunc.interpole( m_Buffer->getData2(_lX-1,_lY), m_Buffer->getData2(_lX,_lY) );

		lVertTable[6] = m_map->newCell(VERTEX);
		m_positions[lVertTable[6]] = recalPoint(vPos,typename PFP::VEC3(interp, 1., 1.));
		m_Buffer->setPointEdge6(_lX, _lY,lVertTable[6]);
	}
}


template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createPointEdge7(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ, unsigned int* const lVertTable, const typename PFP::VEC3& vPos)
{
// TODO parametre _LZ not used => a supprimer ?

	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 128)
	{
		float interp = m_windowFunc.interpole( m_Buffer->getData2(_lX,_lY-1), m_Buffer->getData2(_lX,_lY) );

		lVertTable[7] = m_map->newCell(VERTEX);
		m_positions[lVertTable[7]] = recalPoint(vPos,typename PFP::VEC3(0., interp, 1.));
		m_Buffer->setPointEdge7(_lX, _lY,lVertTable[7]);
	}
}



template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createPointEdge8(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ, unsigned int* const lVertTable, const typename PFP::VEC3& vPos)
{
// TODO parametre _LZ not used => a supprimer ?

	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 256)
	{
		float interp = m_windowFunc.interpole( m_Buffer->getData2(_lX,_lY), m_Buffer->getData2(_lX,_lY) );

		lVertTable[8] = m_map->newCell(VERTEX);
		m_positions[lVertTable[8]] = recalPoint(vPos,typename PFP::VEC3(0., 0., interp));
		m_Buffer->setPointEdge8(_lX, _lY,lVertTable[8]);
	}
}


template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createPointEdge9(const unsigned char _ucCubeIndex,  const int _lX, const int _lY, const int _lZ, unsigned int* const lVertTable, const typename PFP::VEC3& vPos)
{
// TODO parametre _LZ not used => a supprimer ?

	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 512)
	{
		float interp = m_windowFunc.interpole( m_Buffer->getData2(_lX,_lY), m_Buffer->getData2(_lX,_lY+1) );

		lVertTable[9] = m_map->newCell(VERTEX);
		m_positions[lVertTable[9]] = recalPoint(vPos,typename PFP::VEC3(1., 0., interp));
		m_Buffer->setPointEdge9(_lX, _lY,lVertTable[9]);
	}
}


template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createPointEdge10(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ, unsigned int* const lVertTable, const typename PFP::VEC3& vPos)
{
// TODO parametre _LZ not used => a supprimer ?

	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 1024)
	{
		float interp = m_windowFunc.interpole( m_Buffer->getData(_lX,_lY), m_Buffer->getData2(_lX,_lY) );

		lVertTable[10] = m_map->newCell(VERTEX);
		m_positions[lVertTable[10]] = recalPoint(vPos,typename PFP::VEC3(1., 1., interp));
		m_Buffer->setPointEdge10(_lX, _lY,lVertTable[10]);
	}
}


template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createPointEdge11(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ, unsigned int* const lVertTable, const typename PFP::VEC3& vPos)
{
// TODO parametre _LZ not used => a supprimer ?

	if  (accelMCTable::m_EdgeTable[_ucCubeIndex] & 2048)
	{
		float interp = m_windowFunc.interpole( m_Buffer->getData(_lX,_lY), m_Buffer->getData2(_lX,_lY) );

		lVertTable[11] = m_map->newCell(VERTEX);
		m_positions[lVertTable[11]] = recalPoint(vPos,typename PFP::VEC3(0., 1., interp));
		m_Buffer->setPointEdge11(_lX, _lY,lVertTable[11]);
	}
}





template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createFaces_1(const int _lX,const int _lY,const int _lZ, unsigned char tag)
{
	// compute image value and store in buffer
	m_Buffer->setData(_lX, _lY,     m_Image->getVoxel(_lX, _lY,_lZ));
	m_Buffer->setData(_lX+1, _lY,   m_Image->getVoxel(_lX+1, _lY,_lZ));
	m_Buffer->setData(_lX+1, _lY+1, m_Image->getVoxel(_lX+1, _lY+1,_lZ));
	m_Buffer->setData(_lX, _lY+1,   m_Image->getVoxel(_lX, _lY+1 ,_lZ));
	m_Buffer->setData2(_lX, _lY,     m_Image->getVoxel(_lX, _lY,_lZ+1));
	m_Buffer->setData2(_lX+1, _lY,   m_Image->getVoxel(_lX+1, _lY,_lZ+1));
	m_Buffer->setData2(_lX+1, _lY+1, m_Image->getVoxel(_lX+1, _lY+1,_lZ+1));
	m_Buffer->setData2(_lX, _lY+1,   m_Image->getVoxel(_lX, _lY+1 ,_lZ+1));

	unsigned char ucCubeIndex = computeIndex(_lX, _lY);
	if ((ucCubeIndex == 0) || (ucCubeIndex == 255))
		return;

	unsigned int lVertTable[12];

	typename PFP::VEC3 vPos((float)_lX + 0.5f, (float)_lY + 0.5f, (float)_lZ + 0.5f);

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




template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createFaces_2(const int _lX, const int _lY, const int _lZ, unsigned char tag)
{
	// compute image value and store in buffer
	m_Buffer->setData(_lX+1, _lY,   m_Image->getVoxel(_lX+1, _lY,_lZ));
	m_Buffer->setData(_lX+1, _lY+1, m_Image->getVoxel(_lX+1, _lY+1,_lZ));
	m_Buffer->setData2(_lX+1, _lY,   m_Image->getVoxel(_lX+1, _lY,_lZ+1));
	m_Buffer->setData2(_lX+1, _lY+1, m_Image->getVoxel(_lX+1, _lY+1,_lZ+1));

	unsigned char ucCubeIndex = computeIndex(_lX, _lY);
	if ((ucCubeIndex == 0) || (ucCubeIndex == 255))
		return;

	unsigned int lVertTable[12];
//	typename PFP::VEC3 vPos((float)_lX , (float)_lY, (float)_lZ);
	typename PFP::VEC3 vPos((float)_lX + 0.5f, (float)_lY + 0.5f, (float)_lZ + 0.5f);

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



template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createFaces_3(const int _lX, const int _lY, const int _lZ, unsigned char tag)
{
	// compute image value and store in buffer
	m_Buffer->setData(_lX+1, _lY+1, m_Image->getVoxel(_lX+1, _lY+1,_lZ));
	m_Buffer->setData(_lX, _lY+1,   m_Image->getVoxel(_lX, _lY+1 ,_lZ));
	m_Buffer->setData2(_lX+1, _lY+1, m_Image->getVoxel(_lX+1, _lY+1,_lZ+1));
	m_Buffer->setData2(_lX, _lY+1,   m_Image->getVoxel(_lX, _lY+1 ,_lZ+1));

	unsigned char ucCubeIndex = computeIndex(_lX, _lY);
	if ((ucCubeIndex == 0) || (ucCubeIndex == 255))
		return;

	unsigned int lVertTable[12];
//	typename PFP::VEC3 vPos((float)_lX , (float)_lY, (float)_lZ);
	typename PFP::VEC3 vPos((float)_lX + 0.5f, (float)_lY + 0.5f, (float)_lZ + 0.5f);

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


template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createFaces_4(const int _lX, const int _lY, const int _lZ, unsigned char tag)
{
	// compute image value and store in buffer
	m_Buffer->setData(_lX+1, _lY+1, m_Image->getVoxel(_lX+1, _lY+1,_lZ));
	m_Buffer->setData2(_lX+1, _lY+1, m_Image->getVoxel(_lX+1, _lY+1,_lZ+1));

	unsigned char ucCubeIndex = computeIndex(_lX, _lY);
	if ((ucCubeIndex == 0) || (ucCubeIndex == 255))
		return;

	unsigned int lVertTable[12];
//	typename PFP::VEC3 vPos((float)_lX , (float)_lY, (float)_lZ);
	typename PFP::VEC3 vPos((float)_lX + 0.5f, (float)_lY + 0.5f, (float)_lZ + 0.5f);

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


template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createFaces_5(const int _lX, const int _lY, const int _lZ, unsigned char tag)
{
	// compute image value and store in buffer
	m_Buffer->setData2(_lX, _lY,     m_Image->getVoxel(_lX, _lY,_lZ+1));
	m_Buffer->setData2(_lX+1, _lY,   m_Image->getVoxel(_lX+1, _lY,_lZ+1));
	m_Buffer->setData2(_lX+1, _lY+1, m_Image->getVoxel(_lX+1, _lY+1,_lZ+1));
	m_Buffer->setData2(_lX, _lY+1,   m_Image->getVoxel(_lX, _lY+1 ,_lZ+1));

	unsigned char ucCubeIndex = computeIndex(_lX, _lY);
	if ((ucCubeIndex == 0) || (ucCubeIndex == 255))
		return;

	unsigned int lVertTable[12];
//	typename PFP::VEC3 vPos((float)_lX , (float)_lY, (float)_lZ);
	typename PFP::VEC3 vPos((float)_lX + 0.5f, (float)_lY + 0.5f, (float)_lZ + 0.5f);

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


template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createFaces_6(const int _lX, const int _lY, const int _lZ, unsigned char tag)
{
	// compute image value and store in buffer
	m_Buffer->setData2(_lX+1, _lY,   m_Image->getVoxel(_lX+1, _lY,_lZ+1));
	m_Buffer->setData2(_lX+1, _lY+1, m_Image->getVoxel(_lX+1, _lY+1,_lZ+1));

	unsigned char ucCubeIndex = computeIndex(_lX, _lY);
	if ((ucCubeIndex == 0) || (ucCubeIndex == 255))
		return;

	unsigned int lVertTable[12];
//	typename PFP::VEC3 vPos((float)_lX , (float)_lY, (float)_lZ);
	typename PFP::VEC3 vPos((float)_lX + 0.5f, (float)_lY + 0.5f, (float)_lZ + 0.5f);

// create the new  vertices
	int lX = _lX+1;
	int lY = _lY;
	int lZ = _lZ;

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


template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createFaces_7(const int _lX, const int _lY, const int _lZ, unsigned char tag)
{
	// compute image value and store in buffer
	m_Buffer->setData2(_lX+1, _lY+1, m_Image->getVoxel(_lX+1, _lY+1,_lZ+1));
	m_Buffer->setData2(_lX, _lY+1,   m_Image->getVoxel(_lX, _lY+1 ,_lZ+1));

	unsigned char ucCubeIndex = computeIndex(_lX, _lY);
	if ((ucCubeIndex == 0) || (ucCubeIndex == 255))
		return;

	unsigned int lVertTable[12];
//	typename PFP::VEC3 vPos((float)_lX , (float)_lY, (float)_lZ);
	typename PFP::VEC3 vPos((float)_lX + 0.5f, (float)_lY + 0.5f, (float)_lZ + 0.5f);

// create the new  vertices
	int lX = _lX+1;
	int lY = _lY+1;
	int lZ = _lZ;

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



template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createFaces_8(const int _lX, const int _lY, const int _lZ, unsigned char tag)
{
	// compute image value and store in buffer
 	m_Buffer->setData2(_lX+1, _lY+1, m_Image->getVoxel(_lX+1, _lY+1,_lZ+1));

	unsigned char ucCubeIndex = computeIndex(_lX, _lY);
	if ((ucCubeIndex == 0) || (ucCubeIndex == 255))
		return;

	unsigned int lVertTable[12];
//	typename PFP::VEC3 vPos((float)_lX , (float)_lY, (float)_lZ);
	typename PFP::VEC3 vPos((float)_lX + 0.5f, (float)_lY + 0.5f, (float)_lZ + 0.5f);

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


template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::setNeighbourSimple(L_DART d1, L_DART d2)
{
//	d1->set_involution(0,d2);

	if (m_map->phi2(d1) != d2)
		m_map->sewFaces(d1,d2);
}


template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::setNeighbour(L_DART d1, L_DART d2)
{
/*	d1->set_involution(0,d2);
	d2->set_involution(0,d1);*/

	if (m_map->phi2(d1) != d2)
		m_map->sewFaces(d1,d2);
}



template< typename  DataType, typename ImgT, template < typename D2 > class Windowing, class PFP >
void MarchingCubeGen<DataType, ImgT, Windowing, PFP>::createLocalFaces(const unsigned char _ucCubeIndex, const int _lX, const int _lY, const int _lZ,  unsigned int const *_lVertTable, const unsigned short _usMask, float curv, unsigned char tag)
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
	const int8* cTriangle = accelMCTable::m_TriTable[_ucCubeIndex];
	const int8* cNeighbour = accelMCTable::m_NeighTable[_ucCubeIndex];

//	L_DART dartTriangles[5];

// PASS 1: create faces
 	for (int i=0; cTriangle[i]!=-1;/* incrementation inside */)
	{
	// get index of the 3 points of trianglesfrzr
		int lIndexP1 =  cTriangle[i++];
		int lIndexP2 =  cTriangle[i++];
		int lIndexP3 =  cTriangle[i++];

		L_DART edge =createTriEmb(_lVertTable[lIndexP1],  //first point/edge
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
				dn = m_map->phi1(dn);
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
		edge = m_map->phi1(edge);
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
				dn = m_map->phi1(dn);
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
		edge = m_map->phi1(edge);
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
				dn = m_map->phi1(dn);
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

		edge = m_map->phi1(edge); // next edge
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

		edge = m_map->phi1(edge); // next edge
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


// CGoGNout << "NB TRIS: "<< lNumFaces <<CGoGNendl;
// for (int i=0; i< lNumFaces; ++i)
// {
// 	L_DART d = lFacesTab[i];
// 	L_DART dd = L_MAP::phi2(d);
// 	L_DART ddd = L_MAP::phi2(dd);
// 	CGoGNout << d->getLabel()<< " == "<< dd->getLabel()<< " == "<< ddd->getLabel()<< CGoGNendl;
//
//
// 	d = m_map->phi1(d);
// 	dd = L_MAP::phi2(d);
// 	ddd = L_MAP::phi2(dd);
// 	CGoGNout << d->getLabel()<< " == "<< dd->getLabel()<< " == "<< ddd->getLabel()<< CGoGNendl;
//
// 	d = m_map->phi1(d);
// 	dd = L_MAP::phi2(d);
// 	ddd = L_MAP::phi2(dd);
// 	CGoGNout << d->getLabel()<< " == "<< dd->getLabel()<< " == "<< ddd->getLabel()<< CGoGNendl;
//
// }



// finish buffer table of faces with -1
// PAS FORCEMENT UTILE A VERIFIER
// 	for(int i=lNumFaces; i <5; i++)
// 	{
// 		lFacesTab[i] = m_map->end();
// 	}

	#undef EVENMASK
	#undef ODDMASK

}

}
} // end namespace
} // end namespace
} // end namespace

