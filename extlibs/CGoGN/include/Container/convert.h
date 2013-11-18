#ifndef _CONVERT_H_
#define _CONVERT_H_

#include <limits>
#include <math.h>
#include <iostream>

namespace CGoGN
{

/**
 * Conversion d'un T* en U* en fonction de la surcharge des membres
 * Classe abstraite, a surcharger:
 * - reserve(), convert() & destructeur (exemple dans ConvertSimpleCast)
 * 
 */
class ConvertAttrib
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
	 * buffer
	 */
	void* m_buffer;
	
public:
	/**
	 * size of buffer in
	 */
	 ConvertAttrib() : m_size(0),m_nb(0),m_buffer(NULL) {}

	/**
	 * size of buffer in
	 */
	virtual ~ConvertAttrib() {}

	/**
	 * set the number of element to convert & reserve memory for buffer
	 */
	virtual void reserve(unsigned int nb) = 0;

	/**
	 * release memory buffer ( and set ptr to null)
	 */
	virtual void release();

	/**
	 * convert a table to tbe buffer
	 */
	virtual void convert(const void* ptrIn) = 0;

	/**
	 * get the buffer (void*)
	 */
	void* buffer() { return m_buffer; }

	/**
	 * get the size of buffer in bytes
	 */
	unsigned int sizeBuffer() { return m_size; }

	/**
	 * get the size of buffer in bytes
	 */
	unsigned int sizeElt() { return m_size/m_nb; }

	/**
	 * get the size of buffer in bytes
	 */
	unsigned int nbElt() { return m_nb; }
};


/**
* Convertit simplement en castant chaque element
*/
template<typename TYPE_IN, typename TYPE_OUT>
class ConvertSimpleCast : public ConvertAttrib
{
protected:
	TYPE_OUT* m_typedBuffer; 

public:
	~ConvertSimpleCast()
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
		}
	}
	
	void reserve(unsigned int nb)
	{
		m_nb = nb;							// store number of elements
		m_typedBuffer = new TYPE_OUT[nb];	// allocate buffer typed (not possible to delete void*)
		m_buffer = m_typedBuffer;			// store void* casted ptr
		m_size = nb*sizeof(TYPE_OUT);		// store size of buffer in bytes
	}
	
	void convert(const void* ptrIn)
	{
		// cast ptr in & out with right type
		const TYPE_IN* typedIn = reinterpret_cast<const TYPE_IN*>(ptrIn);
		TYPE_OUT* typedOut = reinterpret_cast<TYPE_OUT*>(m_buffer);
		// compute conversion
		for (int i = 0; i < m_nb; ++i)
			*typedOut++ = TYPE_OUT(*typedIn++);
	}
};


/**
* Convertit un type scalaire (char, short, int, ...)
* en un autre (float ou double) en le normalisant
* entre 0 et 1
*/
template<typename TYPE_IN, typename TYPE_OUT>
class ConvertNormalized : public ConvertAttrib
{
protected:
	TYPE_OUT* m_typedBuffer;
	
public:
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
		}
	}
	
	void reserve(unsigned int nb)
	{
		m_nb = nb;							// store number of elements
		m_typedBuffer = new TYPE_OUT[nb];	// allocate buffer
		m_buffer = m_typedBuffer;			// store void* casted ptr
		m_size = nb*sizeof(TYPE_OUT);		// store size of buffer in bytes
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
};


/**
* Convertit un type scalaire (char, short, int, float ..)
* en une couleur variant suivant un schema hsv
* Un min et max donne les valeurs servant à l'interpolation
*/
template<typename TYPE_IN>
class ConvertFloatToRGBf : public ConvertAttrib
{
protected:
	float* m_typedBuffer;
	TYPE_IN m_min;
	TYPE_IN m_diff;

public:
	ConvertFloatToRGBf(TYPE_IN min, TYPE_IN max) : m_min(min), m_diff(max-min) {}
	
	~ConvertFloatToRGBf()
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
};

}

#endif
