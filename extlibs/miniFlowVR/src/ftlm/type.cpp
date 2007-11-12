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
* File: src/ftlm/type.cpp                                         *
*                                                                 *
* Contacts: 20/09/2005 Clement Menier <clement.menier.fr>         *
*                                                                 *
******************************************************************/
#include <ftl/type.h>

#ifdef WIN32
#include <string.h>
#define strcasecmp stricmp
#define atoll atoi
#define snprintf _snprintf
#else
#include <strings.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

namespace ftl
{

namespace Type
{

/// assign specializations

template<>
bool assignString(bool& dest, const std::string& data)
{
  if (data.empty() || !strcasecmp(data.c_str(),"false") || !strcasecmp(data.c_str(),"no"))
  {
    dest = false;
    return true;
  }
  else if (!strcasecmp(data.c_str(),"true") || !strcasecmp(data.c_str(),"yes"))
  {
    dest = true;
    return true;
  }
  else if (data[0] == '-' || (data[0]>='0' && data[0]<='9'))
  {
    dest = (atoi(data.c_str())!=0);
    return true;
  }
  else return false;
}

template<>
bool assignString(char& dest, const std::string& data)
{
  if (data.empty())
  {
    dest = 0;
    return true;
  }
  else if (data[0] == '-' || (data[0]>='0' && data[0]<='9'))
  {
    dest = (char) atoi(data.c_str());
    return true;
  }
  else return false;
}

template<>
bool assignString(unsigned char& dest, const std::string& data)
{
  return assignString((char&)dest, data);
}

template<>
bool assignString(short& dest, const std::string& data)
{
  if (data.empty())
  {
    dest = 0;
    return true;
  }
  else if (data[0] == '-' || (data[0]>='0' && data[0]<='9'))
  {
    dest = (short) atoi(data.c_str());
    return true;
  }
  else return false;
}

template<>
bool assignString(unsigned short& dest, const std::string& data)
{
  return assignString((short&)dest, data);
}

template<>
bool assignString(int& dest, const std::string& data)
{
  if (data.empty())
  {
    dest = 0;
    return true;
  }
  else if (data[0] == '-' || (data[0]>='0' && data[0]<='9'))
  {
    dest = atoi(data.c_str());
    return true;
  }
  else return false;
}

template<>
bool assignString(unsigned int& dest, const std::string& data)
{
  return assignString((int&)dest, data);
}

template<>
bool assignString(long long& dest, const std::string& data)
{
  if (data.empty())
  {
    dest = 0;
    return true;
  }
  else if (data[0] == '-' || (data[0]>='0' && data[0]<='9'))
  {
    dest = atoll(data.c_str());
    return true;
  }
  else return false;
}

template<>
bool assignString(unsigned long long& dest, const std::string& data)
{
  return assignString((long long&)dest, data);
}

template<>
bool assignString(float& dest, const std::string& data)
{
  if (data.empty())
  {
    dest = 0;
    return true;
  }
  else if (data[0] == '-' || (data[0]>='0' && data[0]<='9'))
  {
    dest = (float)atof(data.c_str());
    return true;
  }
  else return false;
}

template<>
bool assignString(double& dest, const std::string& data)
{
  if (data.empty())
  {
    dest = 0;
    return true;
  }
  else if (data[0] == '-' || (data[0]>='0' && data[0]<='9'))
  {
    dest = (double)atof(data.c_str());
    return true;
  }
  else return false;
}

template<>
bool assignString(long double& dest, const std::string& data)
{
  if (data.empty())
  {
    dest = 0;
    return true;
  }
  else if (data[0] == '-' || (data[0]>='0' && data[0]<='9'))
  {
    dest = (long double)atof(data.c_str());
    return true;
  }
  else return false;
}

bool Assign<std::string>::do_assign(std::string& dest, int type, const void* data)
{
  char buf[32];
  if (isSpecial(type))
  {
    if (isString(type))
    {
      dest.assign((const char*)data,size(type));
      return true;
    }
    switch (type)
    {
    case Null:
      dest.clear();
      return true;
    case Enum:
      snprintf(buf,sizeof(buf),"%d",*(const int*)data);
      dest = buf;
      return true;
    case ID:
      snprintf(buf,sizeof(buf),"0x%016llx",*(const long long*)data);
      dest = buf;
      return true;
    }
  }
  else if (isMatrix(type))
  {
    std::string str;
    dest = "{";
    for (int y=0;y<ny(type);y++)
    {
      if (y) dest += ", ";
      dest += '{';
      for (int x=0;x<nx(type);x++)
      {
	if (x) dest += ", ";
	do_assign(str, toSingle(type), data);
	data = ((const char*)data) + elemSize(type);
	dest += str;
      }
      dest += '}';
    }
    dest += '}';
  }
  else if (isVector(type))
  {
    std::string str;
    dest = "{";
    for (int x=0;x<nx(type);x++)
    {
      if (x) dest += ", ";
      do_assign(str, toSingle(type), data);
      data = ((const char*)data) + elemSize(type);
      dest += str;
    }
    dest += '}';
  }
  else
  {
    if (!isSingle(type)) type = toSingle(type);
    switch (toSingle(type))
    {
    case Bool:
      dest = (*(const unsigned char*)data)?"true":"false";
      return true;
    case Byte:
      snprintf(buf,sizeof(buf),"%u",(unsigned)*(const unsigned char*)data);
      dest = buf;
      return true;
    case Short:
      snprintf(buf,sizeof(buf),"%d",(int)*(const short*)data);
      dest = buf;
      return true;
    case Int:
      snprintf(buf,sizeof(buf),"%d",*(const int*)data);
      dest = buf;
      return true;
    case QWord:
      snprintf(buf,sizeof(buf),"%lld",*(const long long*)data);
      dest = buf;
      return true;
    case Float:
      snprintf(buf,sizeof(buf),"%f",(double)*(const float*)data);
      dest = buf;
      return true;
    case Double:
      snprintf(buf,sizeof(buf),"%f",*(const double*)data);
      dest = buf;
      return true;
    case LongDouble:
      snprintf(buf,sizeof(buf),"%Lf",*(const long double*)data);
      dest = buf;
      return true;
    }
  }
  // Failure
  // In this implementation: only Type::Delete is not supported
  return false;
}

// type <-> string

typedef struct {
  int t;
  const char* name;
} TypeToString;

TypeToString names[] =
{ 
  { Delete, "Delete" },
  { Null, "Null" },
  { Enum, "Enum" },
  { ID, "ID" },
  { Bool, "Bool" },
  { Byte, "Byte" },
  { Short, "Short" },
  { Int, "Int" },
  { Long, "Long" },
  { QWord, "QWord" },
  { Half, "Half" },
  { Float, "Float" },
  { Double, "Double" },
  { LongDouble, "LongDouble" },
  { String0, "String" },
  { Null, NULL }
};

std::string name(int t)
{
  char buf[16];
  std::string res;
  int i=0;
  while (names[i].name != NULL && names[i].t != t)
    ++i;
  if (names[i].name != NULL)
  {
    res = names[i].name;
  }
  else if (isMatrix(t))
  {
    res = "Mat<";
    snprintf(buf,sizeof(buf),"%d",ny(t));
    res += buf;
    res += ",";
    snprintf(buf,sizeof(buf),"%d",nx(t));
    res += buf;
    res += ",";
    res += name(toSingle(t));
    res += ">";
  }
  else if (isVector(t))
  {
    res = "Vec<";
    snprintf(buf,sizeof(buf),"%d",nx(t));
    res += buf;
    res += ",";
    res += name(toSingle(t));
    res += ">";
  }
  else if (isString(t))
  {
    if (size(t)==0)
      res = "String";
    else
    {
      res = "String<";
      snprintf(buf,sizeof(buf),"%d",size(t));
      res += buf;
      res += ">";
    }
  }
  else
  {
    snprintf(buf,sizeof(buf),"0x%x",t);
    res = buf;
  }
  return res;
}

int fromName(const std::string& name)
{
  int i=0;
  while (names[i].name != NULL && name != names[i].name)
    ++i;
  if (names[i].name != NULL)
    return names[i].t;
  else if (name.substr(0,7)=="String<")
  {
    int n = 0;
    sscanf(name.c_str(), "String<%d>", &n);
    return buildString(n);
  }
  else if (name.substr(0,4)=="Mat<")
  {
    int nx = 0;
    int ny = 0;
    int pos = 8;
    sscanf(name.c_str(), "Mat<%d,%d,%n", &ny,&nx,&pos);
    int elem = fromName(name.substr(pos,name.length()-1-pos));
    return matrix(elem, nx, ny);
  }
  else if (name.substr(0,4)=="Vec<")
  {
    int nx = 0;
    int pos = 8;
    sscanf(name.c_str(), "Vec<%d,%n", &nx,&pos);
    int elem = fromName(name.substr(pos,name.length()-1-pos));
    return vector(elem, nx);
  }
  else if (name.substr(0,2)=="0x")
  {
    int t = 0;
    sscanf(name.c_str(), "0x%x", &t);
    return t;
  }
  else
  {
    std::cerr << "ERROR converting type from name "<<name<<std::endl;
    return Null;
  }
}


} // namespace Type

} // namespace ftl
