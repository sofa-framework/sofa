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

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MC
{

/*
* constructor
*/
template<typename DART>
Buffer<DART>::Buffer(int _lWidth, int _lHeight)
{
	m_lWidth = _lWidth+1;
	m_lHeight = _lHeight+1;
	m_hcSlice0 = new HalfCube<DART>[m_lWidth*m_lHeight];
	m_hcSlice1 = new HalfCube<DART>[m_lWidth*m_lHeight];
	m_lZpos = 0;

}


/*
* destructor
*/
template<typename DART>
Buffer<DART>::~Buffer()
{                         	
	delete[] m_hcSlice0;
	delete[] m_hcSlice1;
}



/*
* get the face inside a cube of the buffer
*/	
// template <typename DART>
// const DART* Buffer<DART>::getFacesCube(int _lX, int _lY, int _lZ) const
// {
// 
// 	if (_lZ == m_lZpos)				// if current slice ..
// 	{
// 		return m_hcSlice0[_lX + _lY*m_lWidth].m_lFaceEdges;
// 	}
// 	if (_lZ == m_lZpos-1)		// if previous slice ...
// 	{
// 		return m_hcSlice1[_lX + _lY*m_lWidth].m_lFaceEdges;
// 	}
// 	
// 	abort();
// 	return NULL; // never reached but remove warning
// }


/*
* set the face inside a cube of the buffer
*/
template <typename DART>
void Buffer<DART>::setFacesCube(int _lX,int _lY, const DART* const _lFace)
{
	// get the address of cube's table
	DART* lLocFaces  = m_hcSlice1[_lX + _lY*m_lWidth].m_lFaceEdges;
	
//	for (int i=0; i<4; ++i)
	for (int i=0; i<5; ++i)  // 5!!!
	{
		lLocFaces[i] = _lFace[i];     // copy faces indices
	}
}	


/*
* next slice !!
*/
template <typename DART>
void Buffer<DART>::nextSlice()
{
	 m_lZpos ++;					// increase position
	
	 HalfCube<DART> *lTemp = m_hcSlice0;
	 m_hcSlice0 = m_hcSlice1;     // current slice -> old slice
	 m_hcSlice1 = lTemp;          // old slice -> new slice
}


/*
*  get address of face table for direct access
*/
template <typename DART>
DART* Buffer<DART>::getFacesCubeTableAdr(int _lX,int _lY)
{
	return  m_hcSlice0[_lX + _lY*m_lWidth].m_lFaceEdges;
}


/*
* set point on edges functions
*/

template <typename DART>
void Buffer<DART>::setPointEdge0(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice0[_lX + _lY*m_lWidth].m_lX = _lPoint;
}


template <typename DART>
void Buffer<DART>::setPointEdge3(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice0[_lX +(_lY-1)*m_lWidth].m_lY = _lPoint;
}


template <typename DART>
void Buffer<DART>::setPointEdge8(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice0[_lX + _lY*m_lWidth].m_lZ = _lPoint;
}


template <typename DART>
void Buffer<DART>::setPointEdge2(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice0[(_lX-1) +_lY*m_lWidth].m_lX =_lPoint;
}


template <typename DART>
void Buffer<DART>::setPointEdge11(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice0[_lX + _lY*m_lWidth].m_lZ = _lPoint;
}


template <typename DART>
void Buffer<DART>::setPointEdge1(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice0[_lX +_lY*m_lWidth].m_lY =_lPoint;
}


template <typename DART>
void Buffer<DART>::setPointEdge9(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice0[_lX +_lY*m_lWidth].m_lZ =_lPoint;
}


template <typename DART>
void Buffer<DART>::setPointEdge10(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice0[_lX +_lY*m_lWidth].m_lZ =_lPoint;
}


template <typename DART>
void Buffer<DART>::setPointEdge7(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice1[_lX +(_lY-1)*m_lWidth].m_lY =_lPoint;
}


template <typename DART>
void Buffer<DART>::setPointEdge4(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice1[_lX +_lY*m_lWidth].m_lX =_lPoint;
}


template <typename DART>
void Buffer<DART>::setPointEdge6(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice1[(_lX-1) + _lY*m_lWidth].m_lX =_lPoint;
}


template <typename DART>
void Buffer<DART>::setPointEdge5(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice1[_lX +_lY*m_lWidth].m_lY =_lPoint;
}


/*
* get point on edges functions
*/

template <typename DART>
unsigned int Buffer<DART>::getPointEdge0(int _lX,int _lY)
{
	return m_hcSlice0[_lX + _lY*m_lWidth].m_lX;
}


template <typename DART>
unsigned int Buffer<DART>::getPointEdge3(int _lX,int _lY)
{
	return m_hcSlice0[_lX +_lY*m_lWidth].m_lY;
}


template <typename DART>
unsigned int Buffer<DART>::getPointEdge8(int _lX,int _lY)
{
	return m_hcSlice0[_lX + _lY*m_lWidth].m_lZ;
}


template <typename DART>
unsigned int Buffer<DART>::getPointEdge2(int _lX,int _lY)
{
	return m_hcSlice0[_lX +(_lY+1)*m_lWidth].m_lX;
}


template <typename DART>
unsigned int Buffer<DART>::getPointEdge11(int _lX,int _lY)
{
	return m_hcSlice0[_lX + (_lY+1)*m_lWidth].m_lZ;
}


template <typename DART>
unsigned int Buffer<DART>::getPointEdge1(int _lX,int _lY)
{
	return m_hcSlice0[(_lX+1) +_lY*m_lWidth].m_lY;
}


template <typename DART>
unsigned int Buffer<DART>::getPointEdge9(int _lX,int _lY)
{
	return m_hcSlice0[(_lX+1) +_lY*m_lWidth].m_lZ;
}


template <typename DART>
unsigned int Buffer<DART>::getPointEdge10(int _lX,int _lY)
{
	return m_hcSlice0[(_lX+1) +(_lY+1)*m_lWidth].m_lZ;
}


template <typename DART>
unsigned int Buffer<DART>::getPointEdge7(int _lX,int _lY)
{
	return m_hcSlice1[_lX +_lY*m_lWidth].m_lY;
}


template <typename DART>
unsigned int Buffer<DART>::getPointEdge4(int _lX,int _lY)
{
	return m_hcSlice1[_lX +_lY*m_lWidth].m_lX;
}


template <typename DART>
unsigned int Buffer<DART>::getPointEdge6(int _lX,int _lY)
{
	return m_hcSlice1[_lX +(_lY+1)*m_lWidth].m_lX;
}


template <typename DART>
unsigned int Buffer<DART>::getPointEdge5(int _lX,int _lY)
{
	return m_hcSlice1[(_lX+1) +_lY*m_lWidth].m_lY;
}


template <typename DART>
DART Buffer<DART>::getExternalNeighbour(char _cEdge, int _lX, int _lY) const
{

	switch(_cEdge)
	{
		case 0:
			return m_hcSlice0[_lX + _lY*m_lWidth].m_lNX;
			break;
			
		case 1:
			return m_hcSlice0[(_lX+1) +_lY*m_lWidth].m_lNY;
 			break;

		case 2:
			return m_hcSlice0[_lX +(_lY+1)*m_lWidth].m_lNX;
			break;

		case 3:
			return m_hcSlice0[_lX +_lY*m_lWidth].m_lNY;
			break;

		case 4:
			return m_hcSlice1[_lX +_lY*m_lWidth].m_lNX;
			break;

		case 5:
			return m_hcSlice1[(_lX+1) +_lY*m_lWidth].m_lNY;
			break;

		case 6:
			return m_hcSlice1[_lX +(_lY+1)*m_lWidth].m_lNX;
			break;

		case 7:
			return m_hcSlice1[_lX +_lY*m_lWidth].m_lNY;
			break;

		case 8:
			return m_hcSlice0[_lX + _lY*m_lWidth].m_lNZ;
			break;

		case 9:
			return m_hcSlice0[(_lX+1) +_lY*m_lWidth].m_lNZ;
			break;

		case 10:
			return m_hcSlice0[(_lX+1) +(_lY+1)*m_lWidth].m_lNZ;
			break;

		case 11:
			return m_hcSlice0[_lX + (_lY+1)*m_lWidth].m_lNZ;
			break;
	
		default:
			CGoGNerr << "ERROR"<<CGoGNendl;
			return m_hcSlice0[0].m_lNX; // pipo value
	}

	// just for removing the warning never reached because of the default of switch
CGoGNerr << "ERROR"<<CGoGNendl;
	return m_hcSlice0[0].m_lNX;
}


template <typename DART>
void Buffer<DART>::setExternalNeighbour(char _cEdge, int _lX, int _lY, DART _lNeighbour)
{
	switch(_cEdge)
	{
		case 0:
			m_hcSlice0[_lX + _lY*m_lWidth].m_lNX = _lNeighbour;
			break;

		case 1:
			m_hcSlice0[(_lX+1) +_lY*m_lWidth].m_lNY = _lNeighbour;
 			break;

		case 2:
			m_hcSlice0[_lX +(_lY+1)*m_lWidth].m_lNX = _lNeighbour;
			break;

		case 3:
			m_hcSlice0[_lX +_lY*m_lWidth].m_lNY = _lNeighbour;
			break;

		case 4:
			m_hcSlice1[_lX +_lY*m_lWidth].m_lNX = _lNeighbour;
			break;

		case 5:
			m_hcSlice1[(_lX+1) +_lY*m_lWidth].m_lNY = _lNeighbour;
			break;

		case 6:
			m_hcSlice1[_lX +(_lY+1)*m_lWidth].m_lNX = _lNeighbour;
			break;

		case 7:
			m_hcSlice1[_lX +_lY*m_lWidth].m_lNY = _lNeighbour;
			break;

		case 8:
			m_hcSlice0[_lX + _lY*m_lWidth].m_lNZ = _lNeighbour;
			break;

		case 9:
			m_hcSlice0[(_lX+1) +_lY*m_lWidth].m_lNZ = _lNeighbour;
			break;

		case 10:
			m_hcSlice0[(_lX+1) +(_lY+1)*m_lWidth].m_lNZ = _lNeighbour;
			break;

		case 11:
			m_hcSlice0[_lX + (_lY+1)*m_lWidth].m_lNZ = _lNeighbour;
			break;
	
		default:
			CGoGNerr << "ERROR"<<CGoGNendl;
	}
}

}
} // end namespace
} // end namespace
} // end namespace

