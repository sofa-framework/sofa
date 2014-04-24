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

#ifndef __CGoGN_MEM_SHARED_
#define __CGoGN_MEM_SHARED_


#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdlib.h>
#include <iostream>


namespace CGoGN
{

namespace Utils
{
/**
* Shared Memory Segment management
* Example:
*  in master:
*    SharedMem<MY_DATA> shm;
*    shm.initMaster(4321);
*    ...
*     *(shm.lockForWrite()) = localData;
*     shm.release();
*    ...
*  in slave
*    SharedMem<MY_DATA> shm;
*    shm.initSlace(4321);
*    ...
*	 MY_DATA localData = *(shm.dataRead());
*    ...
*/
template<typename DATA>
class SharedMemSeg
{
protected:
	int m_mem_id;
	int* m_ptr;
	DATA* m_ptr1;
	DATA* m_ptr2;
public:
	
	/**
	* Constructor
	*/
	SharedMemSeg();

	/**
	* Destructor
	*/
	~SharedMemSeg();
	
	/**
	* Initialization for master
	* @param key key of shared mem zone
	*/
	bool initMaster(int key);

	/**
	* Initialization for master
	* @param key key of shared mem zone
	*/
	bool initSlave(int key);

	/**
	* read data 
	* @return a pointer on data (for copy or read)
	*/
	DATA* dataRead();
	
	/**
	* lock data for writing. 
	* @return a pointer on data (back buffer) to copy data in
	*/	
	DATA* lockForWrite();

	/**
	* release the data ( switch ptr from back to front)
	*/
	void release();

};

}
}

#include "Utils/shared_mem.hpp"

#endif



