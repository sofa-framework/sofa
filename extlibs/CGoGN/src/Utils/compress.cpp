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

#include <cassert>
#include "Utils/compress.h"
#include "zlib.h"

#include <iostream>
#include <vector>
#include <string.h>
#include <algorithm>

namespace CGoGN
{

namespace Utils
{

void zlibVTUWriteCompressed( unsigned char* input, unsigned int nbBytes, std::ofstream& fout)
{

	std::cout << "Compressor Block " << std::endl;

	const int CHUNK=1024*256;
	int level = 6; // compression level 

	z_stream strm;
	strm.zalloc = Z_NULL;
	strm.zfree = Z_NULL;
	strm.opaque = Z_NULL;
	int ret = deflateInit(&strm, level);
	assert(ret == Z_OK); 

	unsigned char* bufferOut = new unsigned char[nbBytes];
	unsigned char* ptrBufferOut = bufferOut;

	std::vector<unsigned int> header;
	header.reserve(1024);
	header.resize(3);

	unsigned char* ptrData = input;
	int remain = nbBytes;

	while (remain >0)
	{
		strm.avail_in = std::min(remain,CHUNK);	// taille buffer
		strm.next_in = ptrData;					// ptr buffer
		do
		{
			strm.avail_out = CHUNK;
			strm.next_out = ptrBufferOut;
			if (remain>= CHUNK)
				ret = deflate(&strm, 0);
			else
				ret = deflate(&strm, 1);
			assert(ret != Z_STREAM_ERROR); 
			unsigned int have = CHUNK - strm.avail_out;
			ptrBufferOut+=have;
			header.push_back(have);
		} while (strm.avail_out == 0);

		remain -= CHUNK;
		ptrData += CHUNK;	
	}
	deflateEnd(&strm);

	header[0] = header.size()-3;
	header[1] = CHUNK;
	if (remain != 0)
		header[2] = remain +CHUNK;
	else header[2] = 0;

	std::cout << "HEADER "<< std::endl;
	for (unsigned int i=0; i<header.size(); ++i)
		std::cout << header[i]<< std::endl;

	fout.write((char*)&header[0], header.size()*sizeof(unsigned int));
	fout.write((char*)bufferOut, ptrBufferOut - bufferOut);

	std::cout << "DATA "<<  ptrBufferOut - bufferOut << "bytes writen"<< std::endl;

	delete[] bufferOut;
}


}
}
