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
#ifndef __BARRIER_THREAD__
#define __BARRIER_THREAD__


#include <boost/thread/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/thread/condition_variable.hpp>


namespace CGoGN
{

namespace Utils
{

/**
* Implementation of simple counter barrier (rdv)
* for boost::thread
*/
class Barrier
{
private:
	unsigned int m_initCount;
	unsigned int m_count;
	unsigned int m_generation;
	
	boost::mutex m_protect;
	boost::condition_variable  m_cond;

public:

	/**
	* constructor
	* @param count number of threads to syncronize
	*/
	inline Barrier(unsigned int count):
		m_initCount(count), m_count(count), m_generation(0) {}
	
	inline bool wait()
	{
                boost::unique_lock<boost::mutex> lock(m_protect);
		unsigned int gen = m_generation;
		if (--m_count == 0)
		{
			m_generation++;
			m_count = m_initCount;
			m_cond.notify_all();
			return true;
		}

		while (gen == m_generation)
			m_cond.wait(lock);
		return false;
	}
};

}
}

#endif
