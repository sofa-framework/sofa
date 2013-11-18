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
#ifndef _CGOGN_CHRONO_H_
#define _CGOGN_CHRONO_H_


#if defined WIN32
	#include <windows.h>
#else
	#include <sys/time.h>
#endif


namespace CGoGN
{
namespace Utils
{
#if defined _WIN32
class Chrono
{
	ULARGE_INTEGER m_begin;
	ULARGE_INTEGER m_end;
public:
	/// start the chrono
	inline void start()
	{
		FILETIME ttmp={0,0};
		::GetSystemTimeAsFileTime(&ttmp);
	    m_begin.HighPart=ttmp.dwHighDateTime;
	    m_begin.LowPart=ttmp.dwLowDateTime;
	 }

	/// return elapsed time since start (cumulative if several calls)
	inline int elapsed()
	{
		FILETIME ttmp={0,0};
		::GetSystemTimeAsFileTime(&ttmp);
		m_end.HighPart=ttmp.dwHighDateTime;
		m_end.LowPart=ttmp.dwLowDateTime;
    	return int((double(m_end.QuadPart-m_begin.QuadPart)/10000.0));
    }
};
#else
/**
 * class for each time measuring
 */
class Chrono
{
	struct timeval m_start;
	struct timeval m_end;
public:
	/// start the chrono
	inline void start() { gettimeofday(&m_start, NULL) ; }

	/// return elapsed time since start in ms (cumulative if several calls)
	inline int elapsed()
	{
		gettimeofday(&m_end, NULL) ;
	    long seconds  = m_end.tv_sec  - m_start.tv_sec;
	    long useconds = m_end.tv_usec - m_start.tv_usec;
    	return int ((seconds * 1000 + useconds/1000.0) + 0.5);
    }

};
#endif

}
}




#endif /* _CGOGN_CHRONO_H_ */
