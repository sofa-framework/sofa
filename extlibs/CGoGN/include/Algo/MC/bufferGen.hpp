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
template<typename DART, typename DATATYPE>
BufferGen<DART,DATATYPE>::BufferGen(int _lWidth, int _lHeight)
{
	m_lWidth = _lWidth+1;
	m_lHeight = _lHeight+1;
	m_hcSlice0 = new HalfCubeGen<DART>[m_lWidth*m_lHeight];
	m_hcSlice1 = new HalfCubeGen<DART>[m_lWidth*m_lHeight];
	m_lZpos = 0;

	m_dataSl0 = new DATATYPE[m_lWidth*m_lHeight];
	m_dataSl1 = new DATATYPE[m_lWidth*m_lHeight];	

}


/*
* destructor
*/
template<typename DART, typename DATATYPE>
BufferGen<DART,DATATYPE>::~BufferGen()
{                         	
	delete[] m_hcSlice0;
	delete[] m_hcSlice1;

	delete[] m_dataSl0;
	delete[] m_dataSl1;

}



/*
* get the face inside a cube of the buffer
*/	
// template <typename DART, typename DATATYPE>
// const DART* BufferGen<DART,DATATYPE>::getFacesCube(int _lX, int _lY, int _lZ) const
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
template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setFacesCube(int _lX,int _lY, const DART* const _lFace)
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
template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::nextSlice()
{
	 m_lZpos ++;					// increase position
	
	 HalfCubeGen<DART> *lTemp = m_hcSlice0;
	 m_hcSlice0 = m_hcSlice1;     // current slice -> old slice
	 m_hcSlice1 = lTemp;          // old slice -> new slice

	DATATYPE *lTemp2 = m_dataSl0;
	m_dataSl0 = m_dataSl1;
	m_dataSl1 = lTemp2;
}


/*
*  get address of face table for direct access
*/
template <typename DART, typename DATATYPE>
DART* BufferGen<DART,DATATYPE>::getFacesCubeTableAdr(int _lX,int _lY)
{
	return  m_hcSlice0[_lX + _lY*m_lWidth].m_lFaceEdges;
}


/*
* set point on edges functions
*/

template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setPointEdge0(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice0[_lX + _lY*m_lWidth].m_lX = _lPoint;
}


template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setPointEdge3(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice0[_lX +(_lY-1)*m_lWidth].m_lY = _lPoint;
}


template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setPointEdge8(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice0[_lX + _lY*m_lWidth].m_lZ = _lPoint;
}


template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setPointEdge2(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice0[(_lX-1) +_lY*m_lWidth].m_lX =_lPoint;
}


template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setPointEdge11(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice0[_lX + _lY*m_lWidth].m_lZ = _lPoint;
}


template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setPointEdge1(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice0[_lX +_lY*m_lWidth].m_lY =_lPoint;
}


template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setPointEdge9(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice0[_lX +_lY*m_lWidth].m_lZ =_lPoint;
}


template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setPointEdge10(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice0[_lX +_lY*m_lWidth].m_lZ =_lPoint;
}


template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setPointEdge7(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice1[_lX +(_lY-1)*m_lWidth].m_lY =_lPoint;
}


template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setPointEdge4(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice1[_lX +_lY*m_lWidth].m_lX =_lPoint;
}


template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setPointEdge6(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice1[(_lX-1) + _lY*m_lWidth].m_lX =_lPoint;
}


template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setPointEdge5(int _lX,int _lY, unsigned int _lPoint)
{
	m_hcSlice1[_lX +_lY*m_lWidth].m_lY =_lPoint;
}


/*
* get point on edges functions
*/

template <typename DART, typename DATATYPE>
unsigned int BufferGen<DART,DATATYPE>::getPointEdge0(int _lX,int _lY)
{
	return m_hcSlice0[_lX + _lY*m_lWidth].m_lX;
}


template <typename DART, typename DATATYPE>
unsigned int BufferGen<DART,DATATYPE>::getPointEdge3(int _lX,int _lY)
{
	return m_hcSlice0[_lX +_lY*m_lWidth].m_lY;
}


template <typename DART, typename DATATYPE>
unsigned int BufferGen<DART,DATATYPE>::getPointEdge8(int _lX,int _lY)
{
	return m_hcSlice0[_lX + _lY*m_lWidth].m_lZ;
}


template <typename DART, typename DATATYPE>
unsigned int BufferGen<DART,DATATYPE>::getPointEdge2(int _lX,int _lY)
{
	return m_hcSlice0[_lX +(_lY+1)*m_lWidth].m_lX;
}


template <typename DART, typename DATATYPE>
unsigned int BufferGen<DART,DATATYPE>::getPointEdge11(int _lX,int _lY)
{
	return m_hcSlice0[_lX + (_lY+1)*m_lWidth].m_lZ;
}


template <typename DART, typename DATATYPE>
unsigned int BufferGen<DART,DATATYPE>::getPointEdge1(int _lX,int _lY)
{
	return m_hcSlice0[(_lX+1) +_lY*m_lWidth].m_lY;
}


template <typename DART, typename DATATYPE>
unsigned int BufferGen<DART,DATATYPE>::getPointEdge9(int _lX,int _lY)
{
	return m_hcSlice0[(_lX+1) +_lY*m_lWidth].m_lZ;
}


template <typename DART, typename DATATYPE>
unsigned int BufferGen<DART,DATATYPE>::getPointEdge10(int _lX,int _lY)
{
	return m_hcSlice0[(_lX+1) +(_lY+1)*m_lWidth].m_lZ;
}


template <typename DART, typename DATATYPE>
unsigned int BufferGen<DART,DATATYPE>::getPointEdge7(int _lX,int _lY)
{
	return m_hcSlice1[_lX +_lY*m_lWidth].m_lY;
}


template <typename DART, typename DATATYPE>
unsigned int BufferGen<DART,DATATYPE>::getPointEdge4(int _lX,int _lY)
{
	return m_hcSlice1[_lX +_lY*m_lWidth].m_lX;
}


template <typename DART, typename DATATYPE>
unsigned int BufferGen<DART,DATATYPE>::getPointEdge6(int _lX,int _lY)
{
	return m_hcSlice1[_lX +(_lY+1)*m_lWidth].m_lX;
}


template <typename DART, typename DATATYPE>
unsigned int BufferGen<DART,DATATYPE>::getPointEdge5(int _lX,int _lY)
{
	return m_hcSlice1[(_lX+1) +_lY*m_lWidth].m_lY;
}


template <typename DART, typename DATATYPE>
DART BufferGen<DART,DATATYPE>::getExternalNeighbour(char _cEdge, int _lX, int _lY) const
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


template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setExternalNeighbour(char _cEdge, int _lX, int _lY, DART _lNeighbour)
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

/*template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setData(int _lX, int _lY, DATATYPE data)
{
	m_dataSl0[_lX +_lY*m_lWidth] = data;
}


template <typename DART, typename DATATYPE>
DATATYPE BufferGen<DART,DATATYPE>::getData(int _lX, int _lY)
{
	return m_dataSl0[_lX +_lY*m_lWidth];
}

template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setData2(int _lX, int _lY, DATATYPE data)
{
		m_dataSl1[_lX +_lY*m_lWidth] = data;
}


template <typename DART, typename DATATYPE>
DATATYPE BufferGen<DART,DATATYPE>::getData2(int _lX, int _lY)
{
	return m_dataSl1[_lX +_lY*m_lWidth];
}*/


template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setData(int _lX, int _lY, DATATYPE data)
{
//	m_hcSlice0[_lX +_lY*m_lWidth].m_data = data;
	m_dataSl0[_lX +_lY*m_lWidth] = data;
}


template <typename DART, typename DATATYPE>
DATATYPE BufferGen<DART,DATATYPE>::getData(int _lX, int _lY)
{
// 	if (m_dataSl0[_lX +_lY*m_lWidth] != m_hcSlice0[_lX +_lY*m_lWidth].m_data)
// 		CGoGNout << "ERRORRRRR !! "<<CGoGNendl;

	return m_dataSl0[_lX +_lY*m_lWidth];
//	return m_hcSlice0[_lX +_lY*m_lWidth].m_data;
}

template <typename DART, typename DATATYPE>
void BufferGen<DART,DATATYPE>::setData2(int _lX, int _lY, DATATYPE data)
{
		//m_hcSlice1[_lX +_lY*m_lWidth].m_data = data;
		m_dataSl1[_lX +_lY*m_lWidth] = data;
}


template <typename DART, typename DATATYPE>
DATATYPE BufferGen<DART,DATATYPE>::getData2(int _lX, int _lY)
{
// 	if (m_dataSl1[_lX +_lY*m_lWidth] != m_hcSlice1[_lX +_lY*m_lWidth].m_data)
// 		CGoGNout << "ERRORRRRR !! "<<CGoGNendl;
	return m_dataSl1[_lX +_lY*m_lWidth];
//	return m_hcSlice1[_lX +_lY*m_lWidth].m_data;
}


} // end namespace
} // end namespace
} // end namespace
}

