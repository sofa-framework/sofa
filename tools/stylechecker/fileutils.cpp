#include <stdio.h>
#include "fileutils.h"

bool copyFile(const std::string& sourcef, const std::string& destf)
{
    char buf[BUFSIZ];
    size_t size;

    FILE* source = fopen(sourcef.c_str(), "rb");
    FILE* dest = fopen(destf.c_str(), "wb");

    if(!source)
        return false;

    if(!dest){
        fclose(source);
        return false;
    }

    while (size = fread(buf, 1, BUFSIZ, source)) {
        if(fwrite(buf, 1, size, dest) != size){
            fclose(source);
            fclose(dest);
            return false;
        }
    }

    fclose(source);
    fclose(dest);
    return true;
}

