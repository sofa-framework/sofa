/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef OPENCLPROGRAM_H
#define OPENCLPROGRAM_H

#include <string>
#include <cstdio>
#include <stdlib.h>
#include "OpenCLProgramParser.h"
#include "myopencl.h"

namespace sofa
{

namespace helper
{

class OpenCLProgram
{
    cl_program _program;

    std::string  _source,_inputs,_operation;
    std::map<std::string,std::string> _types;

public:
    OpenCLProgram() {}
    OpenCLProgram(std::string * source) {createProgram(source);}
    OpenCLProgram(std::string *source,std::map<std::string,std::string> *types) {createProgram(source,types); }
    OpenCLProgram(std::string *source,std::string *operation)   {createProgram(source,operation);}
    OpenCLProgram(std::string *source,std::string *operation, std::map<std::string,std::string> *types) {createProgram( source,operation,types);}

    ~OpenCLProgram() {if(_program)clReleaseProgram((cl_program)_program); std::cout<<"end\n";}


    void setSource(std::string  s) {_source=s;}
    void setInputs(std::string s) {_inputs=s;}


    void addMacro(std::string name,std::string method)
    {
        if(_source.size()==0) {std::cout << "Error: no source\n"; exit(0);}

        OpenCLProgramParser p(&_source,&method);
        std::string s = p.replaceFunctions(name);

        _source =s;
        createProgram(&_source);
        _source.clear();
        _inputs.clear();
        _operation.clear();
        _types.clear();
    }

    void addMacros(std::string* sources,std::string option)
    {
        OpenCLProgramParser p(&_source);
        _source = p.replaceMacros(sources,option);
    }

    void setTypes(std::map<std::string,std::string> types) {_types=types;}

    void createProgram()
    {
        if(_source.size()==0) {std::cout << "Error: no source\n"; exit(0);}
        if(_types.size()>0)_source = createTypes(&_types) + _source;

        std::cout << "\n--------------\nSOURCE\n---------------\n" << _source << "\n---------\n";

        createProgram(&_source);
    }

    std::string createTypes(std::map<std::string,std::string> *types)
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
        return s;
    }

    void createProgram(std::string * s)
    {
        _program = sofa::gpu::opencl::myopenclProgramWithSource(s->c_str(), s->size());
    }

    void createProgram(std::string *source,std::map<std::string,std::string> *types)
    {
        std::string s;
        std::map<std::string,std::string>::iterator it;

        s = createTypes(types);
        s += *source;
        createProgram(&s);
    }

    void createProgram(std::string *source,std::string *operation)
    {
        OpenCLProgramParser p(source,operation);
        std::string s = p.replaceFunctions();
        createProgram(&s);
    }

    void createProgram(std::string *source,std::string *operation, std::map<std::string,std::string> *types)
    {
        OpenCLProgramParser p(source,operation);
        std::string s = p.replaceFunctions();
        createProgram(&s,types);
    }

    void* program() {return _program;}

    void buildProgram()
    {
        sofa::gpu::opencl::myopenclBuildProgram(_program);
    }

    void buildProgram(char* flags)
    {
        sofa::gpu::opencl::myopenclBuildProgramWithFlags(_program, flags);
    }

    /// fonction qui permet de copier le fichier vers un tableau de caractère
    static std::string* loadSource(const char *file_source)
    {
        std::cout << __FILE__ << " " << __LINE__ << " nom du source:" << file_source << "\n";
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
