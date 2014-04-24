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

#ifndef _CGOGN_FILENAME_H_
#define _CGOGN_FILENAME_H_

#include <fstream>

namespace CGoGN
{
namespace Utils
{
/**
 * @brief check if filename has extension and add it if not
 * @param filename
 * @param extension (with . example ".svg")
 * @return the modified (or not) filename
 */
inline std::string checkFileNameExtension(const std::string &filename, const std::string extension)
{
	std::size_t found = filename.rfind(extension);
	if  ( (found==std::string::npos) || ((found+extension.length()) != filename.length()) )
	{
		if (filename[filename.size()-1]=='.')
			return filename.substr(0,filename.size()-1) + extension;

		return filename + extension;
	}
	return filename;
}

/**
 * @brief extract the path from a file-name
 * @param filename
 * @return the path (with ending /) if there is a / (or \) in filename
 */
inline std::string extractPathFromFileName(const std::string &filename)
{
	std::size_t found = filename.rfind('/');

	if (found == std::string::npos)
		found = filename.rfind('\\'); // welcome on NTFS ;)

	if (found == std::string::npos)
		return "";

	return filename.substr(0,found+1);
}

/**
 * @brief extract the name from a file-name
 * @param filename
 * @return the name of file (string behind last / (or /))
 */
inline std::string extractNameFromFileName(const std::string &filename)
{
	std::size_t found = filename.rfind('/');

	if (found == std::string::npos)
		found = filename.rfind('\\'); // welcome on NTFS ;)

	if (found == std::string::npos)
		return filename;

	return filename.substr(found+1);
}



}
}




#endif

