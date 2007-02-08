#ifndef SOFA_HELPER_IO_IMAGE_H
#define SOFA_HELPER_IO_IMAGE_H

#include <stdlib.h>
#include <sofa/helper/Factory.h>

namespace sofa
{

namespace helper
{

namespace io
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

    int getWidth() const                  { return width; }
    int getHeight() const                 { return height; }
    int getNbBits() const                 { return nbBits; }
    int getLineSize() const               { return ((nbBits+7)/8)*width; }
    int getImageSize() const              { return getLineSize()*height; }
    unsigned char * getData()             { return data; }
    const unsigned char * getData() const { return data; }

    typedef Factory<std::string, Image, std::string> Factory;

    static Image* Create(std::string filename);
};

} // namespace io

} // namespace helper

} // namespace sofa

#endif
