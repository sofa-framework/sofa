#ifndef OPENCLPROGRAM_H
#define OPENCLPROGRAM_H

#include <string>
#include "OpenCLProgramParser.h"
#include "myopencl.h"
#include <CL/cl.h>


namespace sofa
{

namespace helper
{

class OpenCLProgram
{
    cl_program _program;

public:

    OpenCLProgram(std::string * source) {createProgram(source);}
    OpenCLProgram(std::string *source,std::map<std::string,std::string> *types) {createProgram(source,types); }
    OpenCLProgram(std::string *source,std::string *operation)   {createProgram(source,operation);}
    OpenCLProgram(std::string *source,std::string *operation, std::map<std::string,std::string> *types) {createProgram( source,operation,types);}

    ~OpenCLProgram() {if(_program)clReleaseProgram((cl_program)_program); std::cout<<"end\n";}

    void createProgram(std::string * s)
    {
        _program = sofa::gpu::opencl::myopenclProgramWithSource(s->c_str(), s->size());
    }

    void createProgram(std::string *source,std::map<std::string,std::string> *types)
    {
        std::string s;
        std::map<std::string,std::string>::iterator it;


        for( it=types->begin() ; it!=types->end(); it++)
        {
            if(it->second=="double")
                s+="#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
        }

        for( it=types->begin() ; it!=types->end(); it++)
        {
            s += "typedef ";
            s += it->second;
            s += " ";
            s += it->first;
            s += ";\n";
        }

        s += *source;

        createProgram(&s);
    }

    void createProgram(std::string *source,std::string *operation)
    {
        OpenCLProgramParser p(source,operation);
        createProgram(p.replaceFunctions());
    }

    void createProgram(std::string *source,std::string *operation, std::map<std::string,std::string> *types)
    {
        OpenCLProgramParser p(source,operation);
        createProgram(p.replaceFunctions(),types);
    }

    void* program() {return _program;}

    void buildProgram()
    {
        sofa::gpu::opencl::myopenclBuildProgram(_program);
    }

    /// fonction qui permet de copier le fichier vers un tableau de caractère
    static std::string* loadSource(const char *file_source)
    {
        std::string file_name = sofa::gpu::opencl::myopenclPath();
        file_name+= file_source;

        std::string * source = new std::string();
        size_t size;

        //ouvrir le fichier
        FILE *file = fopen(file_name.c_str() ,"rb");
        //si le fichier ne peut pas être ouvert
        if(file==NULL)
        {
            std::cout << "Error: " <<file_name << " could not be opened\n";
            exit(1);
        }

        //chercher la taille du fichier
        fseek(file, 0, SEEK_END);
        size = ftell(file);
        fseek(file,0, SEEK_SET);

        //allouer la taille nécessaire pour que le tableau puisse accueillir le contenu du fichier
        //source = (char*) malloc((*size + 1)*sizeof(char));
        source->resize(size);

        //lire le fichier
        if (fread(&source->at(0), 1, size, file) ==0) {fclose(file); return NULL;}

        fclose(file);
        return source;

    }

    std::string buildLog(int device)
    {
        char *string;
        size_t size;

        clGetProgramBuildInfo((cl_program)_program, (cl_device_id)sofa::gpu::opencl::myopencldevice(device), CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
        string = (char*)malloc(sizeof(char)*(size+1));
        clGetProgramBuildInfo((cl_program)_program,   (cl_device_id)sofa::gpu::opencl::myopencldevice(device), CL_PROGRAM_BUILD_LOG, size, string, NULL);

        string[size]='\0';

        std::string log;
        log +=  "\n=================\n[BEGIN] PROGRAM_BUILD_LOG\n ================\n";
        log.append(string);
        log += "\n\n ===============\n[END] PROGRAM_BUILD_LOG\n=================\n\n";

        free(string);

        return log;
    }


    std::string source()
    {
        char *string;
        size_t size;

        clGetProgramInfo((cl_program)_program, CL_PROGRAM_SOURCE, 0, NULL, &size);
        string = (char*)malloc(sizeof(char)*(size+1));
        clGetProgramInfo((cl_program)_program, CL_PROGRAM_SOURCE, size, string, NULL);

        string[size]='\0';

        std::string log(string);

        free(string);

        return log;
    }

    std::string sourceLog()
    {
        std::string log;

        log += "\n=================\n[BEGIN] PROGRAM_SOURCE\n ================\n\n";
        log += source();
        log += "\n\n ===============\n[END] PROGRAM_SOURCE\n=================\n\n";

        return log;
    }

};

}

}


#endif // OPENCLPROGRAM_H
