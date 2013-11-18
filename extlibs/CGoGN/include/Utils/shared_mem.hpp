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

namespace Utils
{


template<typename DATA>
SharedMemSeg<DATA>::SharedMemSeg():
m_ptr(NULL) 
{ }

template<typename DATA>
SharedMemSeg<DATA>::~SharedMemSeg()
{ 
	if (m_ptr !=NULL)
	{
		struct shmid_ds buf;
		shmctl(m_mem_id,IPC_RMID, &buf);
		shmdt(m_ptr);
	}
}

template<typename DATA>
bool SharedMemSeg<DATA>::initMaster(int key)
{
	m_mem_id = shmget(key, 2*sizeof(DATA)+2*sizeof(int), IPC_CREAT|0666);
	if (m_mem_id < 0)
	{
		CGoGNerr << "Error shmget"<<CGoGNendl;
		return false;
	}

	m_ptr = reinterpret_cast<int*>(shmat(m_mem_id, NULL, 0));
	if (m_ptr ==  (int*)-1)
	{
		CGoGNerr << "Error shmat"<<CGoGNendl;
		return false;
	}

	m_ptr1 = reinterpret_cast<DATA*>(m_ptr + 2);
	m_ptr2 = m_ptr1+1;
	*m_ptr = 0;

	return true ;
}

template<typename DATA>
bool SharedMemSeg<DATA>::initSlave(int key)
{
	m_mem_id = shmget(key, 2*sizeof(DATA)+2*sizeof(int), 0444);
	if (m_mem_id < 0)
	{
		CGoGNerr <<"Shared Memory "<< key << " does not exist can not read"<< CGoGNendl;
		return false;
	}

	m_ptr = reinterpret_cast<int*>(shmat(m_mem_id, NULL, 0));
	if (m_ptr ==  (int*)-1)
	{
		CGoGNerr <<"Problem getting shared memory ptr for "<< key << CGoGNendl;
		return false;
	}
		
	m_ptr1 = reinterpret_cast<DATA*>(m_ptr + 2);
	m_ptr2 = m_ptr1+1;
	*(int*)m_ptr = 0;

	return true ;
}

	
template<typename DATA>
DATA*  SharedMemSeg<DATA>::dataRead()
{
	if (*m_ptr == 0)
		return m_ptr1;
	return  m_ptr2;
}


template<typename DATA>
DATA*  SharedMemSeg<DATA>::lockForWrite()
{
	if (*m_ptr == 0)
		return m_ptr2;
	return m_ptr1;
}

template<typename DATA>
void  SharedMemSeg<DATA>::release()
{
	*m_ptr = 1 - *m_ptr;	//invert id		
}


}
}


