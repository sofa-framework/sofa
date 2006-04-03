#ifndef SOFA_COMPONENTS_COMMON_IMAGE_H
#define SOFA_COMPONENTS_COMMON_IMAGE_H

#include <stdlib.h>
#include "Factory.h"

namespace Sofa
{

namespace Components
{

namespace Common
{

class Image
{
protected:
    int width, height, nbBits;
    unsigned char *data;
public:
    Image();

    virtual ~Image();

    void init(int w, int h, int nbb);
    void clear();

    int getWidth() {return width;}
    int getHeight() {return height;}
    int getNbBits() {return nbBits;}
    unsigned char * getData() {return data;}

    typedef Factory<std::string, Image, std::string> Factory;

    static Image* Create(std::string filename);
};

} // namespace Common

} // namespace Components

} // namespace Sofa

#endif
