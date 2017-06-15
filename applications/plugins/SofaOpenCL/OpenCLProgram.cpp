/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "OpenCLProgram.h"
#include "OpenCLProgramParser.h"

namespace sofa
{

namespace gpu
{

namespace opencl
{

#define DEBUG_TEXT(t) //printf("\t%s\t %s %d\n",t,__FILE__,__LINE__);

OpenCLProgram::OpenCLProgram()
    : _program((cl_program)0)
{}

OpenCLProgram::OpenCLProgram(std::string filename, std::string srcPrefix)
    : _program((cl_program)0)
{
    _filename = filename;
    _source = srcPrefix;
    if (loadSource(_filename, &_source))
        createProgram(&_source);
}

OpenCLProgram::OpenCLProgram(std::string * source)
    : _program((cl_program)0)
{
    createProgram(source);
}

OpenCLProgram::OpenCLProgram(std::string *source,std::map<std::string,std::string> *types)
    : _program((cl_program)0)
{
    createProgram(source,types);
}

OpenCLProgram::OpenCLProgram(std::string filename, std::string srcPrefix,std::map<std::string,std::string> *types)
    : _program((cl_program)0)
{
    _filename = filename;
    _source = srcPrefix;
    if (loadSource(_filename, &_source))
        createProgram(&_source,types);
}

OpenCLProgram::OpenCLProgram(std::string *source,std::string *operation)
    : _program((cl_program)0)
{
    createProgram(source,operation);
}

OpenCLProgram::OpenCLProgram(std::string *source,std::string *operation, std::map<std::string,std::string> *types)
    : _program((cl_program)0)
{
    createProgram( source,operation,types);
}

OpenCLProgram::~OpenCLProgram()
{
    if(_program) clReleaseProgram((cl_program)_program);
}

void OpenCLProgram::setSourceFile(std::string filename, std::string srcPrefix)
{
    _filename = filename;
    _source = srcPrefix;
    loadSource(_filename, &_source);
}

void OpenCLProgram::addMacro(std::string name,std::string method)
{
    if(_source.size()==0) {std::cerr << "Error: no source\n"; exit(0);}

    OpenCLProgramParser p(&_source,&method);
    std::string s = p.replaceFunctions(name);

    _source =s;
    createProgram(&_source);
    _source.clear();
    _inputs.clear();
    _operation.clear();
    _types.clear();
}

void OpenCLProgram::addMacros(std::string* sources,std::string option)
{
    OpenCLProgramParser p(&_source);
    _source = p.replaceMacros(sources,option);
}

void OpenCLProgram::createProgram()
{
    if(_source.size()==0) {std::cerr << "Error: no source\n"; exit(0);}
    if(_types.size()>0)_source = createTypes(&_types) + _source;

    //std::cout << "\n--------------\nSOURCE\n---------------\n" << _source << "\n---------\n";

    createProgram(&_source);
}

std::string OpenCLProgram::createTypes(std::map<std::string,std::string> *types)
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

void OpenCLProgram::createProgram(std::string * s)
{
    _program = sofa::gpu::opencl::myopenclProgramWithSource(s->c_str(), s->size());
}

void OpenCLProgram::createProgram(std::string *source,std::map<std::string,std::string> *types)
{
    std::string s;
    std::map<std::string,std::string>::iterator it;

    s = createTypes(types);
    s += *source;
    createProgram(&s);
}

void OpenCLProgram::createProgram(std::string *source,std::string *operation)
{
    OpenCLProgramParser p(source,operation);
    std::string s = p.replaceFunctions();
    createProgram(&s);
}

void OpenCLProgram::createProgram(std::string *source,std::string *operation, std::map<std::string,std::string> *types)
{
    OpenCLProgramParser p(source,operation);
    std::string s = p.replaceFunctions();
    createProgram(&s,types);
}

void OpenCLProgram::buildProgram()
{
    if (!sofa::gpu::opencl::myopenclBuildProgram(_program))
        std::cerr << buildLog(0) << std::endl;
}

void OpenCLProgram::buildProgram(char* flags)
{
    sofa::gpu::opencl::myopenclBuildProgramWithFlags(_program, flags);
}

/// fonction qui permet de copier le fichier vers un tableau de caractère
bool OpenCLProgram::loadSource(const std::string& file_source, std::string* dest)
{
    std::cout << "OPENCL: Loading " << file_source << std::endl;
//std::cout << __FILE__ << " " << __LINE__ << " nom du source:" << file_source << "\n";
    std::string file_name = sofa::gpu::opencl::myopenclPath();
    file_name+= file_source;

    size_t pos0 = dest->size();
    size_t size;

    //ouvrir le fichier
    FILE *file = fopen(file_name.c_str() ,"rb");
    //si le fichier ne peut pas être ouvert
    if(file==NULL)
    {
        std::cerr << "OPENCL Error: " <<file_name << " could not be opened"<<std::endl;
        exit(702);
    }

    //chercher la taille du fichier
    fseek(file, 0, SEEK_END);
    size = ftell(file);
    fseek(file,0, SEEK_SET);

    //allouer la taille nécessaire pour que le tableau puisse accueillir le contenu du fichier
    //source = (char*) malloc((*size + 1)*sizeof(char));
    dest->resize(pos0+size);

    //lire le fichier
    if (fread(&dest->at(pos0), 1, size, file) ==0) {fclose(file); dest->resize(pos0); return false;}

    fclose(file);
    return true;
}

std::string OpenCLProgram::buildLog(int device)
{
    char *string;
    size_t size;

    clGetProgramBuildInfo((cl_program)_program, (cl_device_id)sofa::gpu::opencl::myopencldevice(device), CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
    if (size == 0) return "";

    string = (char*)malloc(sizeof(char)*(size+1));
    clGetProgramBuildInfo((cl_program)_program,   (cl_device_id)sofa::gpu::opencl::myopencldevice(device), CL_PROGRAM_BUILD_LOG, size, string, NULL);
    while (size > 0 && (string[size-1] == 0 || string[size-1] == ' ' || string[size-1] == '\n' || string[size-1] == '\r' || string[size-1] == '\t')) --size;
    if (size == 0) { free(string); return ""; }

    string[size]='\0';

    std::string log;
    log +=  "\n=================\n[BEGIN] PROGRAM_BUILD_LOG : ";
    log += _filename;
    log += "\n ================\n";
    log.append(string);
    log += "\n ================\n[END]   PROGRAM_BUILD_LOG\n=================\n\n";

    free(string);

    return log;
}


std::string OpenCLProgram::source()
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

std::string OpenCLProgram::sourceLog()
{
    std::string log;

    log += "\n=================\n[BEGIN] PROGRAM_SOURCE : ";
    log += _filename;
    log += "\n ================\n\n";
    log += source();
    log += "\n\n ===============\n[END] PROGRAM_SOURCE\n=================\n\n";

    return log;
}

}

}

}
