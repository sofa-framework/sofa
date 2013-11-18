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
#include "Topology/generic/genericmap.h"
#include "Algo/Parallel/parallel_foreach.h"


namespace CGoGN
{

namespace Algo
{

namespace Parallel
{

void setNbCore(unsigned int nb)
{
	NBCORES = nb;
}



void foreach_attrib(AttributeContainer& attr_cont, FunctorAttribThreaded& func, unsigned int nbth)
{
	if (nbth == 0)
		nbth = optimalNbThreads();

	std::vector<FunctorAttribThreaded*> funcs;
	funcs.reserve(nbth);

	FunctorAttribThreaded* ptr = func.duplicate();
	bool shared = (ptr == NULL);

	if (shared)
	{
		for (unsigned int i = 0; i < nbth; ++i)
			funcs.push_back(&func);
	}
	else
	{
		funcs.push_back(ptr);
		for (unsigned int i = 1; i < nbth; ++i)
			funcs.push_back(func.duplicate());
	}

	foreach_attrib(attr_cont,funcs);

	if (!shared)
		for (unsigned int i = 0; i < nbth; ++i)
			delete funcs[i];

}

void foreach_attrib(AttributeContainer& attr_cont, std::vector<FunctorAttribThreaded*> funcs)
{
	unsigned int nbth = funcs.size();

	std::vector<unsigned int >* vid = new std::vector<unsigned int>[2*nbth];
	boost::thread** threads = new boost::thread*[nbth];

	for (unsigned int i = 0; i < 2*nbth; ++i)
		vid[i].reserve(SIZE_BUFFER_THREAD);

	// fill each vid buffers with 4096 id
	unsigned int id = attr_cont.begin();
	unsigned int nb = 0;
	unsigned int nbm = nbth*SIZE_BUFFER_THREAD;
	while ((id != attr_cont.end()) && (nb < nbm))
	{
		vid[nb%nbth].push_back(id);
		nb++;
		attr_cont.next(id);
	}


	boost::barrier sync1(nbth+1);
	boost::barrier sync2(nbth+1);
	bool finished=false;
	// lauch threads
	for (unsigned int i = 0; i < nbth; ++i)
		threads[i] = new boost::thread(ThreadFunctionAttrib(funcs[i], vid[i],sync1,sync2, finished,1+i));

	while (id != attr_cont.end())
	{
		for (unsigned int i = nbth; i < 2*nbth; ++i)
			vid[i].clear();

		unsigned int nb = 0;
		while ((id != attr_cont.end()) && (nb < nbm))
		{
			vid[nbth + nb%nbth].push_back(id);
			nb++;
			attr_cont.next(id);
		}

		sync1.wait();
		for (unsigned int i = 0; i < nbth; ++i)
			vid[i].swap(vid[nbth+i]);
		sync2.wait();
	}

	sync1.wait();
	finished = true;
	sync2.wait();

	//wait for all theads to be finished
	for (unsigned int i = 0; i < nbth; ++i)
	{
		threads[i]->join();
		delete threads[i];
	}
	delete[] threads;
	delete[] vid;
}


}
} // end namespaces
}
