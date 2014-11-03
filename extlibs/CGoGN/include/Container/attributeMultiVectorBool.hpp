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
#include <cstring>
#include <vector>
namespace CGoGN
{

class MarkerBool
{
public:
	std::string CGoGNnameOfType() const
	{
		return "MarkerBool";
	}
};



template <>
class AttributeMultiVector<MarkerBool> : public AttributeMultiVectorGen
{
	/**
	* table of blocks of data pointers: vectors!
	*/
	std::vector< unsigned int* > m_tableData;
    typedef std::vector< unsigned int* >::iterator Iterator;
    typedef std::vector< unsigned int* >::const_iterator const_iterator;
public:
	AttributeMultiVector(const std::string& strName, const std::string& strType):
		AttributeMultiVectorGen(strName, strType)
	{
		m_tableData.reserve(1024);
	}

	AttributeMultiVector()
	{
		m_tableData.reserve(1024);
	}

	~AttributeMultiVector() {}

	inline AttributeMultiVectorGen* new_obj()
	{
		AttributeMultiVectorGen* ptr = new AttributeMultiVector<MarkerBool>;
		ptr->setTypeName(m_typeName);
		return ptr;
	}

	/**************************************
	 *       MULTI VECTOR MANAGEMENT      *
	 **************************************/

	void addBlock()
	{
		unsigned int* ptr = new unsigned int[_BLOCKSIZE_/32];
		memset(ptr,0,_BLOCKSIZE_/8);
		m_tableData.push_back(ptr);
//		std::cout << "Marker "<<this->getName()<<" - addBlock"<< std::endl;
	}

	void setNbBlocks(unsigned int nbb)
	{
		if (nbb >= m_tableData.size())
		{
			for (unsigned int i= m_tableData.size(); i <nbb; ++i)
				addBlock();
		}
		else
		{
			for (unsigned int i = m_tableData.size()-1; i>=nbb; --i)
				delete[] m_tableData[i];

			m_tableData.resize(nbb);
		}
	}


	unsigned int getNbBlocks() const
	{
		return m_tableData.size();
	}

//	void addBlocksBefore(unsigned int nbb);

	bool copy(const AttributeMultiVectorGen* atmvg)
	{
		const AttributeMultiVector<MarkerBool>* atmv = dynamic_cast<const AttributeMultiVector<MarkerBool>*>(atmvg);
        assert(atmv != NULL);
		if (atmv == NULL)
		{
			CGoGNerr << "trying to copy attributes of different type" << CGoGNendl;
			return false;
		}
//		if (atmv->m_typeName != m_typeName)
//		{
//			CGoGNerr << "trying to copy attributes with different type names" << CGoGNendl;
//			return false;
//		}

		for (unsigned int i = 0; i < atmv->m_tableData.size(); ++i)
			memcpy(m_tableData[i],atmv->m_tableData[i],_BLOCKSIZE_/32);

		return true;
	}

	bool swap(AttributeMultiVectorGen* atmvg)
	{
		AttributeMultiVector<MarkerBool>* atmv = dynamic_cast<AttributeMultiVector<MarkerBool>*>(atmvg);

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

	bool merge(const AttributeMultiVectorGen& att)
	{
		const AttributeMultiVector<MarkerBool>* attrib = dynamic_cast< const AttributeMultiVector<MarkerBool>* >(&att);
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

        for (const_iterator it = attrib->m_tableData.begin(); it != attrib->m_tableData.end(); ++it)
			m_tableData.push_back(*it);

		return true;
	}

	void clear()
	{
        for (const_iterator it=m_tableData.begin(); it !=m_tableData.end(); ++it)
			delete[] *it;
		m_tableData.clear();
	}

	int getSizeOfType() const
	{
		return sizeof(bool); // ?
	}

    inline bool isAllFalse() {
        for (unsigned int i = 0; i < m_tableData.size(); ++i)
        {
            unsigned int *ptr =m_tableData[i];
            for (unsigned int j=0, max = _BLOCKSIZE_/32u; j< max;++j)
                if((*ptr++) != 0u)
                    return false;
        }
        return true;
    }

	inline void allFalse()
	{
		for (unsigned int i = 0; i < m_tableData.size(); ++i)
		{
			unsigned int *ptr =m_tableData[i];
            for (unsigned int j=0, max = _BLOCKSIZE_/32u; j< max;++j)
                *ptr++ = 0;
		}
	}

    inline void allTrue()
    {
        for (unsigned int i = 0; i < m_tableData.size(); ++i)
        {
            unsigned int *ptr =m_tableData[i];
            for (unsigned int j=0, max = _BLOCKSIZE_/32u ; j < max ; ++j)
                *ptr++ = 0xffffffff;
        }
    }

	/**************************************
	 *             DATA ACCESS            *
	 **************************************/


	inline void setFalse(unsigned int i)
	{
		unsigned int jj = i / _BLOCKSIZE_;
		unsigned int j = i % _BLOCKSIZE_;
        unsigned int x = j/32;
        unsigned int y = j%32;
        unsigned int mask = 1 << y;
		m_tableData[jj][x] &= ~mask;
	}

	inline void setTrue(unsigned int i)
	{
		unsigned int jj = i / _BLOCKSIZE_;
		unsigned int j = i % _BLOCKSIZE_;
		unsigned int x = j/32;
		unsigned int y = j%32;
		unsigned int mask = 1 << y;
		m_tableData[jj][x] |= mask;
	}

	inline void setVal(unsigned int i, bool b)
	{
		unsigned int jj = i / _BLOCKSIZE_;
		unsigned int j = i % _BLOCKSIZE_;
		unsigned int x = j/32;
		unsigned int y = j%32;
		unsigned int mask = 1 << y;

		if (b)
			m_tableData[jj][x] |= mask;
		else
			m_tableData[jj][x] &= ~mask;
	}

	/**
	 * get a const reference on a elt
	 * @param i index of element
	 */
	inline bool operator[](unsigned int i) const
	{
		unsigned int jj = i / _BLOCKSIZE_;
		unsigned int j = i % _BLOCKSIZE_;
		unsigned int x = j/32;
		unsigned int y = j%32;

		unsigned int mask = 1 << y;

		return (m_tableData[jj][x] & mask) != 0;
	}

	/**
	 * Get the addresses of each block of data
	 */
	unsigned int getBlocksPointers(std::vector<void*>& addr, unsigned int& /*byteBlockSize*/) const
	{
		CGoGNerr << "DO NOT USE getBlocksPointers with bool attribute"<< CGoGNendl;
		addr.reserve(m_tableData.size());
		addr.clear();

        for (const_iterator it = m_tableData.begin(); it != m_tableData.end(); ++it)
			addr.push_back(NULL );

		return addr.size();
	}


	/**************************************
	 *          LINES MANAGEMENT          *
	 **************************************/

	inline void initElt(unsigned int id)
	{
		setFalse(id);
	}

	inline void copyElt(unsigned int dst, unsigned int src)
	{
		setVal(dst,this->operator [](src));
	}


	inline void swapElt(unsigned int id1, unsigned int id2)
	{
		bool data = this->operator [](id1);
		setVal(id1,this->operator [](id2));
		setVal(id2,data);
	}

	/**
	* swap two elements in container (useful for compact function)
	* @param src_b  block index of source element
	* @param src_id index in block of source element
	* @param dst_b  block index of destination element
	* @param dst_id index in block of destination element
	*/
	void overwrite(unsigned int src_b, unsigned int src_id, unsigned int dst_b, unsigned int dst_id)
	{
		bool b = (m_tableData[src_b][src_id/32] & (1 << (src_id%32))) != 0;

		unsigned int mask = 1 << (dst_id%32);

		if (b)
			m_tableData[dst_b][dst_id/32] |= mask;
		else
			m_tableData[dst_b][dst_id/32] &= ~mask;

	}

	/**************************************
	 *            SAVE & LOAD             *
	 **************************************/

	/**
	 * Sauvegarde binaire
	 * @param fs filestream
	 * @param id id of mv
	 */
	void saveBin(CGoGNostream& fs, unsigned int id)
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
		nbs[1] = nbs[0] * _BLOCKSIZE_/32;
		fs.write(reinterpret_cast<const char*>(nbs),2*sizeof(unsigned int));

        for (const_iterator ptrIt = m_tableData.begin(); ptrIt!=m_tableData.end(); ++ptrIt)
			fs.write(reinterpret_cast<const char*>(*ptrIt),_BLOCKSIZE_/8);
	}


	/**
	 * lecture binaire
	 * @param fs filestream
	 */
	bool loadBin(CGoGNistream& fs)
	{
		unsigned int nbs[2];
		fs.read(reinterpret_cast<char*>(nbs), 2*sizeof(unsigned int));

		unsigned int nb = nbs[0];

		// load data blocks
		m_tableData.resize(nb);

		for(unsigned int i = 0; i < nb; ++i)
		{
			m_tableData[i] = new unsigned int[_BLOCKSIZE_/32];
			fs.read(reinterpret_cast<char*>(m_tableData[i]),_BLOCKSIZE_/8);
		}

		return true;
	}

	/**
	 * lecture binaire
	 * @param fs filestream
	 */
	virtual void dump(unsigned int i) const
	{
		CGoGNout << this->operator[](i);
	}
};



} // namespace CGoGN

