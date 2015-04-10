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

#ifndef _CGOGNSTREAM_H_
#define _CGOGNSTREAM_H_


#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

//forward definitions
namespace CGoGN { namespace Utils { namespace QT { class SimpleQT; } } }
//namespace CGoGN { namespace Utils { namespace QT { class SimpleQGLV; } } }
class QTextEdit;

#include "Utils/dll.h"

namespace CGoGN
{

namespace CGoGNStream
{

/**
 * set all outputs to std
 */
CGoGN_UTILS_API void allToStd(bool yes = true);

/**
 * set all outputs to file
 */
CGoGN_UTILS_API void allToFile(const std::string& filename);

#ifdef CGOGN_WITH_QT
/**
 * set all outputs to status bar of Qt interface
 */
CGoGN_UTILS_API void allToStatusBar(Utils::QT::SimpleQT* sqt);
//CGoGN_UTILS_API void allToStatusBar(Utils::QT::SimpleQGLV* sqglv);

/**
 * set all outputs to console of Qt interface
 */
CGoGN_UTILS_API void allToConsole(Utils::QT::SimpleQT* sqt);
//CGoGN_UTILS_API void allToConsole(Utils::QT::SimpleQGLV* sqglv);

#endif

/**
 * set all outputs to string stream buffer
 */
CGoGN_UTILS_API void allToBuffer(std::stringstream* ss);

enum drawingType
{
	STDOUT = 1,
	STDERR = 2,
	FILEOUT = 4,
	QTSTATUSBAR = 8,
	QTCONSOLE = 16,
	SSBUFFER = 32
};

class Special
{};


class CGoGN_UTILS_API Out
{
protected:
	int m_out_mode;

	std::stringstream m_buffer;
	
#ifdef CGOGN_WITH_QT
	Utils::QT::SimpleQT* m_sqt_bar;
	Utils::QT::SimpleQT* m_sqt_console;

//	Utils::QT::SimpleQGLV* m_sqglv_bar;
//	Utils::QT::SimpleQGLV* m_sqglv_console;


	QTextEdit* m_qte;
#endif

	std::ofstream* m_ofs;

	std::stringstream* m_oss;

	int m_code;

public:
	/**
	 * constructor
	 */
	Out();

	/**
	 * destructor
	 */
	~Out();

	/**
	 * set output to standard
	 */
	void toStd(bool yes = true);

	/**
	 * set output to file
	 */
	void toFile(const std::string& filename);

#ifdef CGOGN_WITH_QT

	/**
	 * remove output to status bars
	 */
	void noStatusBar();

	/**
	 * remove output to consoles
	 */
	void noConsole();

	/**
	 * set output to status bar of Qt interface
	 */
	void toStatusBar(Utils::QT::SimpleQT* sqt = NULL);

	/**
	 * set output to console of Qt interface
	 */
	void toConsole(Utils::QT::SimpleQT* sqt = NULL);

	/**
	 * set output to status bar of Qt interface
	 */
//	void toStatusBar(Utils::QT::SimpleQGLV* sqglv = NULL);

	/**
	 * set output to console of Qt interface
	 */
//	void toConsole(Utils::QT::SimpleQGLV* sqglv = NULL);

#endif

	/**
	 * set output to string stream buffer
	 */
	void toBuffer(std::stringstream* ss);

	/**
	 * recursive stream operator
	 */
	Out& operator<< (Out& cgstr);

	/**
	 * classic stream operator
	 */
	template <typename T>
	Out& operator<< (const T& val);

	/**
	 * special cases (endl) stream operator
	 */
	Out&  operator<< (Special& os  );

	/**
	 * for file closing
	 */
	void close();
};

/**
 * output stream class for error output (replace cout by cerr)
 */
class CGoGN_UTILS_API Err : public Out
{
public:
	Err() { this->m_code = 1; }
};

class Dbg: public Out
{
public:
	Dbg() { this->m_code = 100; }
};


template <typename T>
Out&  Out::operator<< (const T& val)
{
	if (m_out_mode & STDOUT)
		std::cout << val;
	if (m_out_mode & STDERR)
		std::cerr << val;

	if (m_out_mode & (FILEOUT|QTSTATUSBAR|QTCONSOLE|SSBUFFER))
		m_buffer << val;

	return *this;
}

} // namespace CGoGNStream

// glocal stream definitions
CGoGN_UTILS_API extern CGoGNStream::Out CGoGNout;
CGoGN_UTILS_API extern CGoGNStream::Err CGoGNerr;
CGoGN_UTILS_API extern CGoGNStream::Dbg CGoGNdbg;
CGoGN_UTILS_API extern CGoGNStream::Special CGoGNendl;
CGoGN_UTILS_API extern CGoGNStream::Special CGoGNflush;

} // namespace CGoGN

#endif /* CGOGNSTREAM_H_ */
