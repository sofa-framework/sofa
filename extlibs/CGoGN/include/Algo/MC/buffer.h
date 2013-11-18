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

/***************************************************************************
                          buffer.h

Store points and faces of preceding
cubes to optimize algorithm, to share
vertices and compute neighbourhood
                             -------------------
    begin                : Tue Jul 31 2001
    copyright            : (C) 2001 by Sylvain THERY
    email                : 
 ***************************************************************************/

#ifndef BUFFER_H
#define BUFFER_H

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MC
{

/**
* A data structure which store 3 indices (one for each axe) of points and
* a table of faces indices
*
*/


template <typename DART>
class HalfCube
{
public:
	/**
	* index of point on edge X (XY=0)
	*/
	unsigned int m_lX;
	
	/**
	* index of point on edge Y (XZ=0)
	*/
	unsigned int m_lY;
	
	/**
	* index of point on edge Z (XY=0)
	*/
	unsigned int m_lZ;

	/**
	* index of edge of face finishing on edge X (XY=0)
	*/
	DART m_lNX;
	
	/**
	* index of edge of face finishing on edge Y (XZ=0)
	*/
	DART m_lNY;
	
	/**
	* index of edge of face finishing on edge Z (XY=0)
	*/
	DART m_lNZ;	
	
	/**
	* table of edges included in the cube
	*/
	DART m_lFaceEdges[5]; 
};


/**
* @brief Buffer class to speed-up the marching cube
*
* The Buffer class store two slices of HalfCube whose permit
* to have a direct access to points that have been already
* created, and also to faces
*
*/

template < typename DART>
class Buffer
{
protected:
	
	/**
	* first slice of HalfCubes
	*/
	HalfCube<DART> *m_hcSlice0;
	
	/**
	* second slice of HalfCubes
	*/
	HalfCube<DART> *m_hcSlice1;
	
	/**
	* position of the first slice in image
	*/	
	int m_lZpos;

	/**
	* width of slice
	*/	
	int m_lWidth;

	/**
	* heigth of slice
	*/	
	int m_lHeight;
	
public:


	/**
	* constructor
	*************
	* @param   _lWidth the width of image to buffer
	* @param   _lHeight the height of image to buffer
	*/
	Buffer(int _lWidth, int _lHeight);
	
	~Buffer();

	/**
	* @name Function to store the index of a point on edge O..11 of a cube
	* @param  int _lX coordinate of the cube
	* @param  int _lY coordinate of the cube
	* @param  int _lPoint the index of the point
	*/
	//@{
	/**
	*  store the index of a point on edge O..11 of a cube
	* @param  _lX coordinate of the cube
	* @param  _lY coordinate of the cube
	* @param  _lPoint the embedding
	*/
	void setPointEdge0 (int _lX,int _lY, unsigned int _lPoint);
	void setPointEdge1 (int _lX,int _lY, unsigned int _lPoint);
	void setPointEdge2 (int _lX,int _lY, unsigned int _lPoint);
	void setPointEdge3 (int _lX,int _lY, unsigned int _lPoint);
	void setPointEdge4 (int _lX,int _lY, unsigned int _lPoint);
	void setPointEdge5 (int _lX,int _lY, unsigned int _lPoint);
	void setPointEdge6 (int _lX,int _lY, unsigned int _lPoint);
	void setPointEdge7 (int _lX,int _lY, unsigned int _lPoint);
	void setPointEdge8 (int _lX,int _lY, unsigned int _lPoint);
	void setPointEdge9 (int _lX,int _lY, unsigned int _lPoint);
	void setPointEdge10(int _lX,int _lY, unsigned int _lPoint);
	void setPointEdge11(int _lX,int _lY, unsigned int _lPoint);
	//@}

	/**
	* @name Function to get the index of point store on edge O..11 of a cube
	****************************************************
	* @param  int _lX coordinate of the cube
	* @param  int _lY coordinate of the cube
	* @param  int _lPoint the index of the point
	*/
	//@{
	/**
	* get the index of point store on edge O..11 of a cube
	****************************************************
	* @param  _lX coordinate of the cube
	* @param  _lY coordinate of the cube
	* @return   the index of the point
	*/
	unsigned int  getPointEdge0 (int _lX,int _lY);
	unsigned int  getPointEdge1 (int _lX,int _lY);
	unsigned int  getPointEdge2 (int _lX,int _lY);
	unsigned int  getPointEdge3 (int _lX,int _lY);
	unsigned int  getPointEdge4 (int _lX,int _lY);
	unsigned int  getPointEdge5 (int _lX,int _lY);
	unsigned int  getPointEdge6 (int _lX,int _lY);
	unsigned int  getPointEdge7 (int _lX,int _lY);
	unsigned int  getPointEdge8 (int _lX,int _lY);
	unsigned int  getPointEdge9 (int _lX,int _lY);
	unsigned int  getPointEdge10(int _lX,int _lY);
	unsigned int  getPointEdge11(int _lX,int _lY);
	//@}
	
	/**
	* get the address of Face Cube dart Table 
	*************************************
	* @param  _lX x coordinate of cube
	* @param  _lY y coordinate of cube
	* @return   table of faces
	*/
	DART* getFacesCubeTableAdr(int _lX,int _lY);

	/**
	* get dart table of faces stored on edge O..11 of a cube
	*
	* @param   _lX x coordinate of the cube
	* @param   _lY y coordinate of the cube
	* @param   _lZ z coordinate of the cube
	* @return   table of faces
	*/	
	const DART* getFacesCube(int _lX,int _lY,int _lZ) const;
		
	
	/**
	* store indices of faces  on edge O..11 of a cube
	****************************************************
	* @param   _lX x coordinate of the cube
	* @param   _lY y coordinate of the cube
	* @param   _lFace table of faces index
	*/	
	void setFacesCube(int _lX,int _lY, const DART* const _lFace);

	
	/**
	* swap from slice n to n+1 (Z++)
	*/
	void nextSlice();

	/**
	* set neighbour of faces stored on edge O..11 of a cube
	*
	* @param _cEdge code of the edge
	* @param   _lX x coordinate of the cube
	* @param   _lY y coordinate of the cube
	* @param   _lNeighbour neighbour to store
	*/	

	void setExternalNeighbour(char _cEdge, int _lX, int _lY, DART _lNeighbour);

	/**
	* get neighbour stored on edge O..11 of a cube
	* 
	* @param _cEdge code of the edge
	* @param   _lX x coordinate of the cube
	* @param   _lY y coordinate of the cube
	* @return   the neighbour
	*/	
	DART getExternalNeighbour(char _cEdge, int _lX, int _lY) const ;

};

}
} // end namespace
} // end namespace
} // end namespace

#include "buffer.hpp"

#endif
