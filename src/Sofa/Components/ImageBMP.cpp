#include "ImageBMP.h"
#include <iostream>

namespace Sofa
{

namespace Components
{

using namespace Common;

Creator<Image::Factory,ImageBMP> ImageBMPClass("bmp");

void ImageBMP::init(const std::string &filename)
{
    unsigned short int bfType;
    long int bfOffBits;
    short int biPlanes;
    short int biBitCount;
    long int biSizeImage;
    int i;
    unsigned char temp;
    FILE *file;
    /* make sure the file is there and open it read-only (binary) */
    if ((file = fopen(filename.c_str(), "rb")) == NULL)
    {
        std::cerr << "File not found : " << filename << std::endl;
        return;
    }

    if(!fread(&bfType, sizeof(short int), 1, file))
    {
        std::cerr << "Error reading file!" << std::endl;
        return;
    }

    /* check if file is a bitmap */
    if (bfType != 19778)
    {
        std::cerr << "Not a Bitmap-File!\n";
        return;
    }
    /* get the file size */
    /* skip file size and reserved fields of bitmap file header */
    fseek(file, 8, SEEK_CUR);
    /* get the position of the actual bitmap data */
    if (!fread(&bfOffBits, sizeof(long int), 1, file))
    {
        std::cerr << "Error reading file!\n";
        return;
    }
    // printf("Data at Offset: %ld\n", bfOffBits);
    /* skip size of bitmap info header */
    fseek(file, 4, SEEK_CUR);
    /* get the width of the bitmap */
    fread(&width, sizeof(int), 1, file);
    //printf("Width of Bitmap: %d\n", texture->width);
    /* get the height of the bitmap */
    fread(&height, sizeof(int), 1, file);
    //printf("Height of Bitmap: %d\n", texture->height);
    /* get the number of planes (must be set to 1) */
    fread(&biPlanes, sizeof(short int), 1, file);
    if (biPlanes != 1)
    {
        std::cerr << "Error: number of Planes not 1!\n";
        return;
    }
    /* get the number of bits per pixel */
    if (!fread(&biBitCount, sizeof(short int), 1, file))
    {
        std::cerr << "Error reading file!\n";
        return;
    }
    //printf("Bits per Pixel: %d\n", biBitCount);
    if (biBitCount != 24)
    {
        std::cerr << "Bits per Pixel not 24\n";
        return;
    }
    /* calculate the size of the image in bytes */
    biSizeImage = width * height * 3;
    // std::cout << "Size of the image data: " << biSizeImage << std::endl;
    data = (unsigned char*) malloc(biSizeImage);
    /* seek to the actual data */
    fseek(file, bfOffBits, SEEK_SET);

    if (!fread(data, biSizeImage, 1, file))
    {
        std::cerr << "Error loading file!\n";
        return;
    }

    /* swap red and blue (bgr -> rgb) */
    for (i = 0; i < biSizeImage; i += 3)
    {
        temp = data[i];
        data[i] = data[i + 2];
        data[i + 2] = temp;
    }
}

} // namespace Components

} // namespace Sofa
