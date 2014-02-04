#ifndef _CONVERT_H_
#define _CONVERT_H_

#include <limits>
#include <math.h>
#include <iostream>
#include "Geometry/vector_gen.h"

namespace CGoGN
{

/**
 * Conversion d'un T* en U* en fonction de la surcharge des membres
 * Classe abstraite, a surcharger:
 * - reserve(), convert(), release() & vectorSize()
 * 
 */
class ConvertBuffer
{
protected:
	/**
	 * size of buffer in bytes
	 */
	unsigned int m_size;
	
	/**
	 * size of buffer in number of elements
	 */
	unsigned int m_nb;

	/**
	 * size of element in input
	 */
	unsigned int m_szElt;

	/**
	 * buffer
	 */
	void* m_buffer;

public:
	/**
	 *
	 */
	ConvertBuffer() : m_size(0),m_nb(0),m_szElt(0),m_buffer(NULL) {}

	ConvertBuffer(unsigned int sz) : m_size(0),m_nb(0),m_szElt(sz),m_buffer(NULL) {}

	/**
	 *
	 */
//	virtual ~ConvertBuffer() = 0;
	 ~ConvertBuffer() {}

	/**
	 * set the number of element to convert & reserve memory for buffer
	 * @param nb nb elements (usually block size when call from VBO)
	 */
	virtual void reserve(unsigned int nb) = 0;

	/**
	 * release memory buffer ( and set ptrs to null)
	 */
	 virtual void release() =  0;

	/**
	 * convert the buffer
	 */
	virtual void convert(const void* ptrIn) = 0;

	/**
	 * get the data size in elements (ex: 3 for Vec3f)
	 */
	virtual unsigned int vectorSize() = 0;

	/**
	 * get the buffer (void*)
	 */
	void* buffer() { return m_buffer; }

	/**
	 * get the size of buffer in bytes
	 */
	unsigned int sizeBuffer() { return m_size; }

};


/**
* Convertit simplement en castant chaque element
* Exemple double ou int float
*/
template<typename TYPE_IN>
class ConvertToFloat : public ConvertBuffer
{
protected:
	float* m_typedBuffer;
	unsigned int m_szVect;

public:
	ConvertToFloat():
		ConvertBuffer(sizeof(TYPE_IN)),m_szVect(1)
	{}

	~ConvertToFloat()
	{
		if (m_typedBuffer)
			delete[] m_typedBuffer;
	}

	void release()
	{
		delete[] m_typedBuffer;
		m_typedBuffer = NULL;
		m_buffer = NULL;
		m_nb = 0;
		m_size = 0;
	}
	
	void reserve(unsigned int nb)
	{
		m_nb = nb*m_szVect;						// store number of elements
		m_typedBuffer = new float[m_nb];	// allocate buffer
		m_buffer = m_typedBuffer;			// store void* casted ptr
		m_size = m_nb*sizeof(float);		// store size of buffer in bytes
	}
	
	void convert(const void* ptrIn)
	{
		// cast ptr in & out with right type
		const TYPE_IN* typedIn = reinterpret_cast<const TYPE_IN*>(ptrIn);
		float* typedOut = reinterpret_cast<float*>(m_buffer);
		// compute conversion
		for (unsigned int i = 0; i < m_nb; ++i)
			*typedOut++ = float(*typedIn++);
	}

	unsigned int vectorSize()
	{
		return m_szVect;
	}

	void setPseudoVectorSize(unsigned int sz)
	{
		m_szVect = sz;
	}
};


/**
* Convertit un type scalaire (char, short, int, ...)
* en un autre (float ou double) en le normalisant
* entre 0 et 1
*/
template<typename TYPE_IN, typename TYPE_OUT>
class ConvertNormalized : public ConvertBuffer
{
protected:
	TYPE_OUT* m_typedBuffer;
	
public:
	ConvertNormalized():
		ConvertBuffer(sizeof(TYPE_IN))
	{}

	~ConvertNormalized()
	{
		if (m_typedBuffer)
			delete[] m_typedBuffer;
	}

	void release()
	{
		if (m_typedBuffer)
		{
			delete[] m_typedBuffer;
			m_typedBuffer = NULL;
			m_buffer = NULL;
			m_nb = 0;
			m_size = 0;
		}
	}
	
	void reserve(unsigned int nb)
	{
		m_nb = nb;							// store number of elements
		m_typedBuffer = new TYPE_OUT[m_nb];	// allocate buffer
		m_buffer = m_typedBuffer;			// store void* casted ptr
		m_size = m_nb*sizeof(TYPE_OUT);		// store size of buffer in bytes
	}
	
	void convert(const void* ptrIn)
	{
		// cast ptr in & out with right type
		const TYPE_IN* typedIn = reinterpret_cast<const TYPE_IN*>(ptrIn);
		TYPE_OUT* typedOut = reinterpret_cast<TYPE_OUT*>(m_buffer);
		
		// compute conversion
		for (int i=0; i <m_nb; ++i)
		{
			TYPE_OUT val = (float(*typedIn++) - TYPE_OUT(std::numeric_limits<TYPE_IN>::min()));
			val /= TYPE_OUT(std::numeric_limits<TYPE_IN>::max()) - TYPE_OUT(std::numeric_limits<TYPE_IN>::min());
			*typedOut++ = val;
		}
	}

	unsigned int vectorSize()
	{
		return 1;
	}

};


/**
* Convertit un type scalaire (char, short, int, float ..)
* en une couleur variant suivant un schema hsv
* Un min et max donne les valeurs servant à l'interpolation
*/
template<typename TYPE_IN>
class ConvertToRGBf : public ConvertBuffer
{
protected:
	float* m_typedBuffer;
	TYPE_IN m_min;
	TYPE_IN m_diff;

public:
	ConvertToRGBf(TYPE_IN min, TYPE_IN max) : ConvertBuffer(sizeof(TYPE_IN)),m_min(min), m_diff(max-min) {}

	~ConvertToRGBf()
	{
		if (m_typedBuffer)
			delete[] m_typedBuffer;
	}

	void release()
	{
		if (m_typedBuffer)
		{
			delete[] m_typedBuffer;
			m_typedBuffer = NULL;
			m_buffer = NULL;
			m_nb = 0;
			m_size = 0;
		}
	}
	
	void reserve(unsigned int nb)
	{
		m_nb = nb;							// store number of elements
		m_typedBuffer = new float[3*nb];	// allocate buffer
		m_buffer = m_typedBuffer;			// store void* casted ptr
		m_size = 3*nb*sizeof(float);		// store size of buffer in bytes
	}
	
	void convert(const void* ptrIn)
	{
		// cast ptr in & out with right type
		const TYPE_IN* typedIn = reinterpret_cast<const TYPE_IN*>(ptrIn);
		float* typedOut = reinterpret_cast<float*>(m_buffer);
		
		// compute conversion
		for (int i=0; i <m_nb; ++i)
		{
			float h = (360.0f /(m_diff))*(*typedIn++ - m_min); // normalize in 0-360
			int hi = int(floor(h / 60.0f)) % 6;
			float f = (h / 60.0) - floor(h / 60.0);
			float q = 1.0f - f;
			switch(hi)
			{
				case 0:
					*typedOut++ = 0.0f;
					*typedOut++ = f;
					*typedOut++ = 1.0f;
					break;
				case 1:
					*typedOut++ = 0.0f;
					*typedOut++ = 1.0f;
					*typedOut++ = q;
					break;
				case 2:
					*typedOut++ = f;
					*typedOut++ = 1.0f;
					*typedOut++ = 0.0f;
					break;
				case 3:
					*typedOut++ = 1.0f;
					*typedOut++ = q;
					*typedOut++ = 0.0f;
					break;
				case 4:
					*typedOut++ = 1.0f;
					*typedOut++ = 0.0f;
					*typedOut++ = f;
					break;
				case 5:
					*typedOut++ = q;
					*typedOut++ = 0.0f;
					*typedOut++ = 1.0f;
				default:
					break;
			}
		}
	}

	unsigned int vectorSize()
	{
		return 1;
	}
};



class ConvertVec3dToVec3f : public ConvertBuffer
{
protected:
	Geom::Vec3f* m_typedBuffer;

public:
	ConvertVec3dToVec3f():
		ConvertBuffer(sizeof(Geom::Vec3d))
	{}

	~ConvertVec3dToVec3f()
	{
		if (m_typedBuffer)
			delete[] m_typedBuffer;
	}

	void release()
	{
		delete[] m_typedBuffer;
		m_typedBuffer = NULL;
		m_buffer = NULL;
		m_nb = 0;
		m_size = 0;
	}

	void reserve(unsigned int nb)
	{
		m_nb = nb;							// store number of elements
		m_typedBuffer = new Geom::Vec3f[nb];	// allocate buffer typed (not possible to delete void*)
		m_buffer = m_typedBuffer;			// store void* casted ptr
		m_size = nb*sizeof(Geom::Vec3f);		// store size of buffer in bytes
	}

	void convert(const void* ptrIn)
	{
		// cast ptr in & out with right type
		const Geom::Vec3d* typedIn = reinterpret_cast<const Geom::Vec3d*>(ptrIn);
		Geom::Vec3f* typedOut = reinterpret_cast<Geom::Vec3f*>(m_buffer);
		// compute conversion
		for (unsigned int i = 0; i < m_nb; ++i)
		{
			const Geom::Vec3d& vd = *typedIn++;
			*typedOut++ = Geom::Vec3f(vd[0],vd[1],vd[2]);
//			Geom::Vec3f& vf = *typedOut++;
//			vf[0]=vd[0];vf[1]=vd[1];vf[2]=vd[2];
		}
	}

	unsigned int vectorSize()
	{
		return 3;
	}
};



typedef ConvertToFloat<double> ConvertDoubleToFloat;

}

#endif
