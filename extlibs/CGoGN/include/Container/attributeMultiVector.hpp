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

inline AttributeMultiVectorGen::AttributeMultiVectorGen(const std::string& strName, const std::string& strType):
	m_attrName(strName), m_typeName(strType)
{}

inline AttributeMultiVectorGen::AttributeMultiVectorGen()
{}

inline AttributeMultiVectorGen::~AttributeMultiVectorGen()
{}

/**************************************
 *             ACCESSORS              *
 **************************************/

inline unsigned int AttributeMultiVectorGen::getOrbit() const
{
	return m_orbit;
}

inline void AttributeMultiVectorGen::setOrbit(unsigned int i)
{
	m_orbit = i ;
}

inline unsigned int AttributeMultiVectorGen::getIndex() const
{
	return m_index;
}

inline void AttributeMultiVectorGen::setIndex(unsigned int i)
{
	m_index = i ;
}

inline const std::string& AttributeMultiVectorGen::getName() const
{
	return m_attrName;
}

inline void AttributeMultiVectorGen::setName(const std::string& n)
{
	m_attrName = n ;
}

inline const std::string& AttributeMultiVectorGen::getTypeName() const
{
	return m_typeName;
}

inline void AttributeMultiVectorGen::setTypeName(const std::string& n)
{
	m_typeName = n ;
}

inline unsigned int AttributeMultiVectorGen::getBlockSize() const
{
	return _BLOCKSIZE_ ;
}


/***************************************************************************************************/
/***************************************************************************************************/


template <typename T>
AttributeMultiVector<T>::AttributeMultiVector(const std::string& strName, const std::string& strType):
	AttributeMultiVectorGen(strName, strType)
{
	m_tableData.reserve(1024);
}

template <typename T>
AttributeMultiVector<T>::AttributeMultiVector()
{
	m_tableData.reserve(1024);
}

template <typename T>
AttributeMultiVector<T>::~AttributeMultiVector()
{
	for (typename std::vector< T* >::iterator it = m_tableData.begin(); it != m_tableData.end(); ++it)
		delete[] (*it);
}

template <typename T>
inline AttributeMultiVectorGen* AttributeMultiVector<T>::new_obj()
{
	AttributeMultiVectorGen* ptr = new AttributeMultiVector<T>;
	ptr->setTypeName(m_typeName);
	return ptr;
}

/**************************************
 *       MULTI VECTOR MANAGEMENT      *
 **************************************/

template <typename T>
inline void AttributeMultiVector<T>::addBlock()
{
	T* ptr = new T[_BLOCKSIZE_];
	m_tableData.push_back(ptr);
	// init
//	T* endPtr = ptr + _BLOCKSIZE_;
//	while (ptr != endPtr)
//		*ptr++ = T(0);
}

template <typename T>
void AttributeMultiVector<T>::setNbBlocks(unsigned int nbb)
{
	if (nbb >= m_tableData.size())
	{
		for (unsigned int i= m_tableData.size(); i <nbb; ++i)
			addBlock();
	}
	else
	{
		for (unsigned int i = nbb; i < m_tableData.size(); ++i)
			delete[] m_tableData[i];
		m_tableData.resize(nbb);
	}
}

template <typename T>
unsigned int AttributeMultiVector<T>::getNbBlocks() const
{
	return m_tableData.size();
}

template <typename T>
void AttributeMultiVector<T>::addBlocksBefore(unsigned int nbb)
{
	std::vector<T*> tempo;
	tempo.reserve(1024);

	for (unsigned int i = 0; i < nbb; ++i)
		addBlock();

	for (typename std::vector<T*>::iterator it = m_tableData.begin(); it != m_tableData.end(); ++it)
		tempo.push_back(*it);

	m_tableData.swap(tempo);
}

template <typename T>
bool AttributeMultiVector<T>::copy(const AttributeMultiVectorGen* atmvg)
{
	const AttributeMultiVector<T>* atmv = dynamic_cast<const AttributeMultiVector<T>*>(atmvg);

	if (atmv == NULL)
	{
		CGoGNerr << "trying to copy attributes of different type" << CGoGNendl;
		return false;
	}
	if (atmv->m_typeName != m_typeName)
	{
		CGoGNerr << "trying to swap attributes with different type names" << CGoGNendl;
		return false;
	}

	for (unsigned int i = 0; i < atmv->m_tableData.size(); ++i)
		std::memcpy(m_tableData[i], atmv->m_tableData[i], _BLOCKSIZE_ * sizeof(T));

	return true;
}

template <typename T>
bool AttributeMultiVector<T>::swap(AttributeMultiVectorGen* atmvg)
{
	AttributeMultiVector<T>* atmv = dynamic_cast<AttributeMultiVector<T>*>(atmvg);

	if (atmv == NULL)
	{
		CGoGNerr << "trying to swap attributes of different type" << CGoGNendl;
		return false;
	}
	if (atmv->m_typeName != m_typeName)
	{
		CGoGNerr << "trying to swap attributes with different type names" << CGoGNendl;
		return false;
	}

	m_tableData.swap(atmv->m_tableData) ;
	return true;
}

template <typename T>
bool AttributeMultiVector<T>::merge(const AttributeMultiVectorGen& att)
{
	const AttributeMultiVector<T>* attrib = dynamic_cast< const AttributeMultiVector<T>* >(&att);
	if (attrib==NULL)
	{
		CGoGNerr << "trying to merge attributes of different type" << CGoGNendl;
		return false;
	}

	if (attrib->m_typeName != m_typeName)
	{
		CGoGNerr << "trying to merge attributes with different type names" << CGoGNendl;
		return false;
	}

	for (typename std::vector<T*>::const_iterator it = attrib->m_tableData.begin(); it != attrib->m_tableData.end(); ++it)
		m_tableData.push_back(*it);

	return true;
}

template <typename T>
inline void AttributeMultiVector<T>::clear()
{
	for (typename std::vector< T* >::iterator it = m_tableData.begin(); it != m_tableData.end(); ++it)
		delete[] (*it);
	m_tableData.clear();
}

template <typename T>
inline int AttributeMultiVector<T>::getSizeOfType() const
{
	return sizeof(T);
}

/**************************************
 *             DATA ACCESS            *
 **************************************/

template <typename T>
inline T& AttributeMultiVector<T>::operator[](unsigned int i)
{
	return m_tableData[i / _BLOCKSIZE_][i % _BLOCKSIZE_];
}

template <typename T>
inline const T& AttributeMultiVector<T>::operator[](unsigned int i) const
{
	return m_tableData[i / _BLOCKSIZE_][i % _BLOCKSIZE_];
}

template <typename T>
unsigned int AttributeMultiVector<T>::getBlocksPointers(std::vector<void*>& addr, unsigned int& byteBlockSize) const
{
	byteBlockSize = _BLOCKSIZE_ * sizeof(T);

	addr.reserve(m_tableData.size());
	addr.clear();

	for (typename std::vector<T*>::const_iterator it = m_tableData.begin(); it != m_tableData.end(); ++it)
		addr.push_back(*it);

	return addr.size();
}

/**************************************
 *          LINES MANAGEMENT          *
 **************************************/

template <typename T>
inline void AttributeMultiVector<T>::initElt(unsigned int id)
{
	m_tableData[id / _BLOCKSIZE_][id % _BLOCKSIZE_] = T(); // T(0);
}

template <typename T>
inline void AttributeMultiVector<T>::copyElt(unsigned int dst, unsigned int src)
{
	m_tableData[dst / _BLOCKSIZE_][dst % _BLOCKSIZE_] = m_tableData[src / _BLOCKSIZE_][src % _BLOCKSIZE_];
}

template <typename T>
void AttributeMultiVector<T>::swapElt(unsigned int id1, unsigned int id2)
{
	T data = m_tableData[id1 / _BLOCKSIZE_][id1 % _BLOCKSIZE_] ;
	m_tableData[id1 / _BLOCKSIZE_][id1 % _BLOCKSIZE_] = m_tableData[id2 / _BLOCKSIZE_][id2 % _BLOCKSIZE_] ;
	m_tableData[id2 / _BLOCKSIZE_][id2 % _BLOCKSIZE_] = data ;
}

template <typename T>
void AttributeMultiVector<T>::overwrite(unsigned int src_b, unsigned int src_id, unsigned int dst_b, unsigned int dst_id)
{
	m_tableData[dst_b][dst_id] = m_tableData[src_b][src_id];
}


/**************************************
 *            SAVE & LOAD             *
 **************************************/

template <typename T>
void AttributeMultiVector<T>::saveBin(CGoGNostream& fs, unsigned int id)
{
	unsigned int nbs[3];
	nbs[0] = id;
	int len1 = m_attrName.size()+1;
	int len2 = m_typeName.size()+1;
	nbs[1] = len1;
	nbs[2] = len2;
	fs.write(reinterpret_cast<const char*>(nbs),3*sizeof(unsigned int));
	// store names
	char buffer[256];
	const char* s1 = m_attrName.c_str();
	memcpy(buffer,s1,len1);
	const char* s2 = m_typeName.c_str();
	memcpy(buffer+len1,s2,len2);
	fs.write(reinterpret_cast<const char*>(buffer),(len1+len2)*sizeof(char));

	nbs[0] = m_tableData.size();
	nbs[1] = nbs[0] * _BLOCKSIZE_* sizeof(T);
	fs.write(reinterpret_cast<const char*>(nbs),2*sizeof(unsigned int));

	// store data blocks
	for(unsigned int i=0; i<nbs[0]; ++i)
	{
		fs.write(reinterpret_cast<const char*>(m_tableData[i]),_BLOCKSIZE_*sizeof(T));
	}
}

inline unsigned int AttributeMultiVectorGen::loadBinInfos(CGoGNistream& fs, std::string& name, std::string& type)
{
	unsigned int nbs[3];
	fs.read(reinterpret_cast<char*>(nbs), 3*sizeof(unsigned int));

	unsigned int id = nbs[0];
	unsigned int len1 = nbs[1];
	unsigned int len2 = nbs[2];

    char* buffer = new char[256];
	fs.read(buffer, (len1+len2)*sizeof(char));

	name = std::string(buffer);
	type = std::string(buffer + len1);

	return id;
}

template <typename T>
bool AttributeMultiVector<T>::loadBin(CGoGNistream& fs)
{
	unsigned int nbs[2];
	fs.read(reinterpret_cast<char*>(nbs), 2*sizeof(unsigned int));

	unsigned int nb = nbs[0];

	// load data blocks
	m_tableData.resize(nb);
	for(unsigned int i = 0; i < nb; ++i)
	{
		T* ptr = new T[_BLOCKSIZE_];
		fs.read(reinterpret_cast<char*>(ptr),_BLOCKSIZE_*sizeof(T));
		m_tableData[i] = ptr;
	}

	return true;
}

inline bool AttributeMultiVectorGen::skipLoadBin(CGoGNistream& fs)
{
	unsigned int nbs[2];
	fs.read(reinterpret_cast<char*>(nbs), 2*sizeof(unsigned int));

	// get number of byte to skip
	unsigned int nbb = nbs[1];

	// check if nbb ok
	if (nbb % _BLOCKSIZE_ != 0)
	{
		CGoGNerr << "Error skipping wrong number of byte in attributes reading"<< CGoGNendl;
		return false;
	}

	// skip data (no seek because of pb with gzstream)
	char* ptr = new char[_BLOCKSIZE_];
	while (nbb != 0)
	{
		nbb -= _BLOCKSIZE_;
		fs.read(reinterpret_cast<char*>(ptr),_BLOCKSIZE_);
	}
	delete[] ptr;

	return true;
}


template <typename T>
void AttributeMultiVector<T>::dump(unsigned int i) const
{
	CGoGNout << this->operator[](i);
}

} // namespace CGoGN
