#include "showvector.h"

ShowVector::ShowVector(char *fileName)
{
    _file = fopen(fileName,"w");
    _name = std::string(fileName);
}

ShowVector::~ShowVector()
{
    fclose(_file);
}

template <>
void ShowVector::writeVector(int v)
{
    fprintf(_file,"%d",v);
}

template <>
void ShowVector::writeVector(float v)
{
    fprintf(_file,"%f",v);
}
