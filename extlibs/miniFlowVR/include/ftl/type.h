/******* COPYRIGHT ************************************************
*                                                                 *
*                             FlowVR                              *
*                       Template Library                          *
*                                                                 *
*-----------------------------------------------------------------*
* COPYRIGHT (C) 20054 by                                          *
* Laboratoire Informatique et Distribution (UMR5132) and          *
* INRIA Project MOVI. ALL RIGHTS RESERVED.                        *
*                                                                 *
* This source is covered by the GNU LGPL, please refer to the     *
* COPYING file for further information.                           *
*                                                                 *
*-----------------------------------------------------------------*
*                                                                 *
*  Original Contributors:                                         *
*    Jeremie Allard,                                              *
*    Clement Menier.                                              *
*                                                                 * 
*******************************************************************
*                                                                 *
* File: include/ftl/type.h                                        *
*                                                                 *
* Contacts: 20/09/2005 Clement Menier <clement.menier.fr>         *
*                                                                 *
******************************************************************/
#ifndef FTL_TYPE_H
#define FTL_TYPE_H

#include <string.h>
#include <string>

namespace ftl
{

namespace Type
{

/// size: 14 bits (total size)
/// elem: 4 bits (size of each element)
/// nx: 5 bits (number of columns-1)
/// ny: 5 bits (number of rows-1)
/// type: 2 bits
///   0 = Special types (empty or delete if elem==0, string if elem==1, enum if elem==4, ID if elem==8)
///   1 = RESERVED
///   2 = Numerical types
///   3 = Floating point types
/// endian: 2 bits
///   00 = Don't care about endianness (empty)
///   01 = Endianness OK
///   10 = Endianness reversed
///   11 = Special types (delete)

/// The current bit allocation is:
/// bit  0 -  1 : type
/// bit  2 -  6 : nx
/// bit  7      : endian 1
/// bit  8 - 12 : ny
/// bit 13 - 16 : elem
/// bit 17 - 30 : size
/// bit 31      : endian 2

enum
{

  MASK_TYPE   = 0x00000003, SHIFT_TYPE   =  0,
  MASK_NX     = 0x0000007C, SHIFT_NX     =  2,
  MASK_NY     = 0x00001F00, SHIFT_NY     =  8,
  MASK_ELEM   = 0x0001E000, SHIFT_ELEM   = 13,
  MASK_SIZE   = 0x7FFE0000, SHIFT_SIZE   = 17,
  MASK_ENDIAN = 0x80000080,
  ENDIAN_OK   = 0x00000080,
  ENDIAN_SWAP = 0x80000000,
  TYPE_SPECIAL  = 0,
  TYPE_RESERVED = 1 << SHIFT_TYPE,
  TYPE_NUM      = 2 << SHIFT_TYPE,
  TYPE_REAL     = 3 << SHIFT_TYPE,

};

// This must be a macro to be able to initialize enum Type

#define FLOWVR_TYPE2(type, elembits, nx, ny) \
  ( ENDIAN_OK | (type) | (((elembits)/8) << SHIFT_ELEM) \
    | (((nx)-1) << SHIFT_NX) | (((ny)-1) << SHIFT_NY) \
    | ((((elembits)*(nx)*(ny)+7)/8) << SHIFT_SIZE) )

#define FLOWVR_TYPE1(type, elembits, nx) FLOWVR_TYPE2(type, elembits, nx, 1)
#define FLOWVR_TYPE(type, elembits) FLOWVR_TYPE2(type, elembits, 1, 1)

extern inline int build(int type, int elembits, int nx=1, int ny=1)
{
  return FLOWVR_TYPE2(type, elembits, nx, ny);
}

extern inline int buildString(int len)
{
  return ENDIAN_OK | TYPE_SPECIAL | (1 << SHIFT_ELEM)
    | ((len) << SHIFT_SIZE);
}

extern inline int endianOk(int t)
{
  return (t & MASK_ENDIAN) != ENDIAN_SWAP;
}

extern inline int type(int t)
{
  return (t & MASK_TYPE);
}

extern inline int elemSize(int t)
{
  return (t & MASK_ELEM) >> SHIFT_ELEM;
}

extern inline int size(int t)
{
  return (t & MASK_SIZE) >> SHIFT_SIZE;
}

extern inline int nx(int t)
{
  return ((t & MASK_NX) >> SHIFT_NX)+1;
}

extern inline int ny(int t)
{
  return ((t & MASK_NY) >> SHIFT_NY)+1;
}

extern inline bool isSpecial(int t)
{
  return type(t) == TYPE_SPECIAL;
}

extern inline bool isNum(int t)
{
  return type(t) == TYPE_NUM;
}

extern inline bool isReal(int t)
{
  return type(t) == TYPE_REAL;
}

extern inline bool isString(int t)
{
  return (t & (MASK_TYPE|MASK_ELEM)) == (TYPE_SPECIAL|(1<<SHIFT_ELEM));
}

extern inline bool isSingle(int t)
{
  return nx(t)==1 && ny(t)==1;
}

extern inline bool isVector(int t)
{
  return nx(t)>1 && ny(t)==1;
}

extern inline bool isMatrix(int t)
{
  return ny(t)>1;
}

extern inline int toSingle(int t)
{
  int e = elemSize(t);
  if (e==0) e=1;
  return ( t & ~(MASK_NX|MASK_NY|MASK_SIZE) ) | (e << SHIFT_SIZE);
}

extern inline int toVector(int t)
{
  return ( t & ~(MASK_NY|MASK_SIZE) ) | ((size(t)/ny(t)) << SHIFT_SIZE);
}

extern inline int vector(int elemtype, int nx)
{
  return (elemtype &~ (MASK_NX|MASK_NY|MASK_SIZE)) | ((nx-1)<<SHIFT_NX) | ((elemSize(elemtype)*nx) << SHIFT_SIZE);
}

extern inline int matrix(int elemtype, int nx, int ny)
{
  return (elemtype &~ (MASK_NX|MASK_NY|MASK_SIZE)) | ((nx-1)<<SHIFT_NX) | ((ny-1)<<SHIFT_NY) | ((elemSize(elemtype)*nx*ny) << SHIFT_SIZE);
}

enum Type
{

  /// @name Special Types
  /// @{

  Delete = MASK_ENDIAN,          ///< Any pre-existing value must be removed or reset
  Null   = 0,                    ///< Empty value
  String0 = FLOWVR_TYPE( TYPE_SPECIAL,  8 ), ///< Text string
  Enum   = FLOWVR_TYPE( TYPE_SPECIAL, 32 ), ///< Special constants
  ID     = FLOWVR_TYPE( TYPE_SPECIAL, 64 ), ///< 64 bits ID

  /// @}

  /// @name Primitive Types
  /// @{

  Bool   = FLOWVR_TYPE( TYPE_NUM ,  1 ),
  Byte   = FLOWVR_TYPE( TYPE_NUM ,  8 ),
  Short  = FLOWVR_TYPE( TYPE_NUM , 16 ),
  Int    = FLOWVR_TYPE( TYPE_NUM , 32 ),
  Long   = FLOWVR_TYPE( TYPE_NUM , 8*sizeof(long) ), // can be either equal to Int or QWord depending on architecture
  QWord  = FLOWVR_TYPE( TYPE_NUM , 64 ),
  Half   = FLOWVR_TYPE( TYPE_REAL, 16 ),
  Float  = FLOWVR_TYPE( TYPE_REAL, 32 ),
  Double = FLOWVR_TYPE( TYPE_REAL, 64 ),
  LongDouble = FLOWVR_TYPE( TYPE_REAL, 96 ),

  /// @}

  /// @name Vector Types
  /// @{

  Vec2b = FLOWVR_TYPE1( TYPE_NUM ,  8, 2 ),  Vec3b = FLOWVR_TYPE1( TYPE_NUM ,  8, 3 ),  Vec4b = FLOWVR_TYPE1( TYPE_NUM ,  8, 4 ),
  Vec2s = FLOWVR_TYPE1( TYPE_NUM , 16, 2 ),  Vec3s = FLOWVR_TYPE1( TYPE_NUM , 16, 3 ),  Vec4s = FLOWVR_TYPE1( TYPE_NUM , 16, 4 ),
  Vec2i = FLOWVR_TYPE1( TYPE_NUM , 32, 2 ),  Vec3i = FLOWVR_TYPE1( TYPE_NUM , 32, 3 ),  Vec4i = FLOWVR_TYPE1( TYPE_NUM , 32, 4 ),
  Vec2q = FLOWVR_TYPE1( TYPE_NUM , 64, 2 ),  Vec3q = FLOWVR_TYPE1( TYPE_NUM , 64, 3 ),  Vec4q = FLOWVR_TYPE1( TYPE_NUM , 64, 4 ),
  Vec2h = FLOWVR_TYPE1( TYPE_REAL, 16, 2 ),  Vec3h = FLOWVR_TYPE1( TYPE_REAL, 16, 3 ),  Vec4h = FLOWVR_TYPE1( TYPE_REAL, 16, 4 ),
  Vec2f = FLOWVR_TYPE1( TYPE_REAL, 32, 2 ),  Vec3f = FLOWVR_TYPE1( TYPE_REAL, 32, 3 ),  Vec4f = FLOWVR_TYPE1( TYPE_REAL, 32, 4 ),
  Vec2d = FLOWVR_TYPE1( TYPE_REAL, 64, 2 ),  Vec3d = FLOWVR_TYPE1( TYPE_REAL, 64, 3 ),  Vec4d = FLOWVR_TYPE1( TYPE_REAL, 64, 4 ),
  Vec2ld= FLOWVR_TYPE1( TYPE_REAL, 96, 2 ),  Vec3ld= FLOWVR_TYPE1( TYPE_REAL, 96, 3 ),  Vec4ld= FLOWVR_TYPE1( TYPE_REAL, 96, 4 ),

  Vec8bits  = FLOWVR_TYPE1( TYPE_NUM ,  1, 8 ),
  Vec16bits = FLOWVR_TYPE1( TYPE_NUM ,  1, 16 ),
  Vec24bits = FLOWVR_TYPE1( TYPE_NUM ,  1, 24 ),
  Vec32bits = FLOWVR_TYPE1( TYPE_NUM ,  1, 32 ),

  /// @}

  /// @name Matrix Types
  /// @{

  Mat2x2f = FLOWVR_TYPE2( TYPE_REAL, 32, 2,2 ),  Mat2x3f = FLOWVR_TYPE2( TYPE_REAL, 32, 2,3 ),  Mat2x4f = FLOWVR_TYPE2( TYPE_REAL, 32, 2,4 ),
  Mat3x2f = FLOWVR_TYPE2( TYPE_REAL, 32, 3,2 ),  Mat3x3f = FLOWVR_TYPE2( TYPE_REAL, 32, 3,3 ),  Mat3x4f = FLOWVR_TYPE2( TYPE_REAL, 32, 3,4 ),
  Mat4x2f = FLOWVR_TYPE2( TYPE_REAL, 32, 4,2 ),  Mat4x3f = FLOWVR_TYPE2( TYPE_REAL, 32, 4,3 ),  Mat4x4f = FLOWVR_TYPE2( TYPE_REAL, 32, 4,4 ),

  Mat2x2d = FLOWVR_TYPE2( TYPE_REAL, 64, 2,2 ),  Mat2x3d = FLOWVR_TYPE2( TYPE_REAL, 64, 2,3 ),  Mat2x4d = FLOWVR_TYPE2( TYPE_REAL, 64, 2,4 ),
  Mat3x2d = FLOWVR_TYPE2( TYPE_REAL, 64, 3,2 ),  Mat3x3d = FLOWVR_TYPE2( TYPE_REAL, 64, 3,3 ),  Mat3x4d = FLOWVR_TYPE2( TYPE_REAL, 64, 3,4 ),
  Mat4x2d = FLOWVR_TYPE2( TYPE_REAL, 64, 4,2 ),  Mat4x3d = FLOWVR_TYPE2( TYPE_REAL, 64, 4,3 ),  Mat4x4d = FLOWVR_TYPE2( TYPE_REAL, 64, 4,4 ),

  /// @}
  
};

template<typename T>
inline Type get(const T&) { return T::getType(); }

template<> inline Type get(const bool&) { return Bool; }
template<> inline Type get(const char&) { return Byte; }
template<> inline Type get(const unsigned char&) { return Byte; }
template<> inline Type get(const short&) { return Short; }
template<> inline Type get(const unsigned short&) { return Short; }
template<> inline Type get(const int&) { return Int; }
template<> inline Type get(const unsigned int&) { return Int; }
template<> inline Type get(const long long&) { return QWord; }
template<> inline Type get(const unsigned long long&) { return QWord; }
template<> inline Type get(const float&) { return Float; }
template<> inline Type get(const double&) { return Double; }
template<> inline Type get(const std::string& s) { return (Type)buildString(s.length()); }

std::string name(int t);
int fromName(const std::string& name);

/// Set a variable to Null (0/empty) value
/// Default implementation: use operator=(0)
template<class T>
bool assignNull(T& dest)
{
  dest = 0;
  return true;
}

/// Assign a variable from a string
/// Default implementation: use operator=
template<class T>
bool assignString(T& dest, const std::string& data)
{
  dest = data;
  return true;
}

/// Assign a variable from an enum valye
/// Default implementation: use operator=
template<class T>
bool assignEnum(T& dest, int data)
{
  dest = data;
  return true;
}

/// Assign a variable from an ID
/// Default implementation: use operator=
template<class T>
bool assignID(T& dest, long long data)
{
  dest = data;
  return true;
}

/// Assign a variable from a typed data.
/// Return false if failed.
/// Default implementation: use constructor from all single-valued types
template<class T>
class Assign
{
public:
  static bool do_assign(T& dest, int type, const void* data)
  {
    if (isSpecial(type))
    {
      if (isString(type))
      {
	std::string str((const char*)data,size(type));
	return assignString(dest,str);
      }
      switch (type)
      {
      case Null: return assignNull(dest);
      case Enum: return assignEnum((int &) dest,*(const int*)data);
      case ID: return assignID((long long &) dest,*(const long long*)data);
      }
    }
    else
    {
      if (!isSingle(type)) type = toSingle(type);
      switch (type)
      {
      case Bool:       dest = (T)(bool)( *(const unsigned char*)data != 0 );
	return true;
      case Byte:       dest = (T)*(const unsigned char*)data;
	return true;
      case Short:      dest = (T)*(const short*)data;
	return true;
      case Int:        dest = (T)*(const int*)data;
	return true;
      case QWord:      dest = (T)*(const long long*)data;
	return true;
      case Float:      dest = (T)*(const float*)data;
	return true;
      case Double:     dest = (T)*(const double*)data;
	return true;
      case LongDouble: dest = (T)*(const long double*)data;
	return true;
      }
    }
    // Failure
    // In this implementation: only Type::Delete is not supported
    return false;
  }
};

template<class T>
bool assign(T& dest, int type, const void* data)
{
    return Assign<T>::do_assign(dest,type,data);
}

// Specializations

template<> bool assignString(bool& dest, const std::string& data);
template<> bool assignString(char& dest, const std::string& data);
template<> bool assignString(unsigned char& dest, const std::string& data);
template<> bool assignString(short& dest, const std::string& data);
template<> bool assignString(unsigned short& dest, const std::string& data);
template<> bool assignString(int& dest, const std::string& data);
template<> bool assignString(unsigned int& dest, const std::string& data);
template<> bool assignString(long long& dest, const std::string& data);
template<> bool assignString(unsigned long long& dest, const std::string& data);
template<> bool assignString(float& dest, const std::string& data);
template<> bool assignString(double& dest, const std::string& data);
template<> bool assignString(long double& dest, const std::string& data);

template<>
class Assign<std::string>
{
public:
  static bool do_assign(std::string& dest, int type, const void* data);
};

} // namespace Type

/// 16 bits word byteswap operation
extern inline unsigned short wswap(unsigned short w)
{
  return (w>>8) | (w<<8);
}

/// 32 bits dword byteswap operation
extern inline unsigned int dswap(unsigned int d)
{
  return (d>>24) | ((d>>8)&0x0000ff00)
    | ((d<<8)&0x00ff0000) | (d<<24);
}

/// 64 bits qword byteswap operation
extern inline unsigned long long qswap(unsigned long long q)
{
  // TODO: better swap...
  return (unsigned long long)(dswap((unsigned int)(q>>32)))
    | (((unsigned long long)(dswap((unsigned int)q)))<<32);
}

/// Array of values of a specified type.
///
/// Note: For vector or matrix types, a value contains several elements.
class TypedArray
{
protected:
  int valueType;
  unsigned int dataSize;
  union
  {
    unsigned char sdata[16]; ///< for small data
    struct
    {
      unsigned char* ptr; ///< for larger data
      int* refs; ///< Number of references to the allocated data
    };
  };
public:

  TypedArray()
  : valueType(Type::Null), dataSize(0), ptr(NULL), refs(NULL)
  {
  }

  TypedArray(int type, const void* data, unsigned int size)
  : valueType(type)
  {
    if (data == NULL || size<=0)
    {
      ptr = NULL;
      refs = NULL;
      dataSize = 0;
    }
    else if (size <= sizeof(sdata))
    {
      dataSize = size;
      memcpy(sdata,data,size);
    }
    else
    {
      ptr = new unsigned char[size];
      memcpy(ptr,data,size);
      dataSize=size;
      refs = new int;
      *refs = 1;
    }
    if (dataSize>0 && !Type::endianOk(valueType))
    {
      /// Must swap endianness
      valueType = dswap(valueType);
      switch (Type::elemSize(valueType))
      {
      case 2:
      {
	unsigned short* p = (unsigned short*)this->data();
	for (unsigned int i=0;i<dataSize/sizeof(short);i++)
	  p[i] = wswap(p[i]);
	break;
      }
      case 4:
      {
	unsigned int* p = (unsigned int*)this->data();
	for (unsigned int i=0;i<dataSize/sizeof(int);i++)
	  p[i] = dswap(p[i]);
	break;
      }
      case 8:
      {
	unsigned long long* p = (unsigned long long*)this->data();
	for (unsigned int i=0;i<dataSize/sizeof(long long);i++)
	  p[i] = qswap(p[i]);
	break;
      }
      // Other element sizes don't need to be swapped
      }
    }
  }

  TypedArray(const TypedArray& from)
  : valueType(from.valueType), dataSize(from.dataSize)
  {
    if (dataSize == 0)
    {
      ptr = NULL;
      refs = NULL;
    }
    else if (dataSize <= sizeof(sdata))
    {
      memcpy(sdata,from.sdata,dataSize);
    }
    else
    {
      ptr = from.ptr;
      refs = from.refs;
      ++(*refs);
    }
  }

  ~TypedArray()
  {
    if (dataSize > sizeof(sdata))
    {
      if (--(*refs)==0)
      {
	delete[] ptr;
	delete refs;
      }
    }
  }

  void operator=(const TypedArray& from)
  {
    if (dataSize > sizeof(sdata))
    {
      if (--(*refs)==0)
      {
	delete[] ptr;
	delete refs;
      }
    }
    valueType=from.valueType;
    dataSize=from.dataSize;
    if (dataSize == 0)
    {
      ptr = NULL;
      refs = NULL;
    }
    else if (dataSize <= sizeof(sdata))
    {
      memcpy(sdata,from.sdata,dataSize);
    }
    else
    {
      ptr = from.ptr;
      refs = from.refs;
      ++(*refs);
    }
  }

  const unsigned char* data() const
  {
    return (dataSize<=sizeof(sdata))?sdata:ptr;
  }

  unsigned int size() const
  {
    return dataSize;
  }

  unsigned int n() const
  {
    unsigned int v = valueSize();
    if (v==0) return 0;
    return size() / v;
  }

  bool empty() const
  {
    return n()==0;
  }

  int type() const
  {
    return valueType;
  }

  bool isSpecial() const
  {
    return Type::isSpecial(valueType);
  }

  bool isNum() const
  {
    return Type::isNum(valueType);
  }

  bool isReal() const
  {
    return Type::isReal(valueType);
  }

  bool isString() const
  {
    return Type::isString(valueType);
  }

  bool isSingle() const
  {
    return Type::isSingle(valueType);
  }

  bool isVector() const
  {
    return Type::isVector(valueType);
  }

  bool isMatrix() const
  {
    return Type::isMatrix(valueType);
  }

  unsigned int valueSize() const
  {
    return Type::size(valueType);
  }

  unsigned int elemSize() const
  {
    return Type::elemSize(valueType);
  }

  int elemType() const
  {
    if (isSingle()) return valueType;
    else return Type::toSingle(valueType);
  }

  unsigned int nx() const
  {
    return Type::nx(valueType);
  }

  unsigned int ny() const
  {
    return Type::ny(valueType);
  }

  template <class T>
  bool assign(T& dest, int n=0) const
  {
    return Type::assign(dest, valueType, data()+valueSize()*n);
  }

  template <class T>
  T get(int n=0) const
  {
    T dest;
    assign<T>(dest, n);
    return dest;
  }
  
};

} // namespace ftl

#endif
