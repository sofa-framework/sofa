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
#ifndef SOFAOPENCL_OPENCLPROGRAMPARSER_H
#define SOFAOPENCL_OPENCLPROGRAMPARSER_H

#include <iostream>
#include <vector>
#include <map>


namespace sofa
{

namespace gpu
{

namespace opencl
{

class OpenCLProgramParser
{
    std::string _program;
    std::string _operation;
    std::map<std::string,std::string> _map;

public:
    OpenCLProgramParser(std::string *program,std::string *operation)
    {
        _program=*program;
        _operation=*operation;
    }

    OpenCLProgramParser(std::string *program)
    {
        _program=*program;
    }

    // find args used by the fonction
    void parseFonction(std::string name = "__OP__")
    {
        //find the name of the fonction
        int begin = _operation.find(name);
        begin = _operation.find("(",begin+name.size());
        int end = _operation.find(")",begin);

        //find args of the fonction
        int i,n;
        int j=begin;
        while(begin<=j && j<end)
        {
            //find args between '(' and ',' OR ',' and ',' OR ',' and ')' OR '(' and ')'
            i=j+1;
            j = _operation.find(",",i);
            if(j==-1)j=end;
            std::string s =_operation.substr(i,j-i);
            //delete spaces characters
            while((n = s.find_first_of(" \t\n"))!=-1)s.erase(n,1);
            //insert arg in map
            _map.insert( std::pair<std::string,std::string>(s,""));
        }

        std::map<std::string,std::string>::iterator it;

        /*		std::cout << "parseFunction\n";
        		// show content:
        		for ( it=_map.begin() ; it != _map.end(); it++ )
        			std::cout << (*it).first << " => " << (*it).second << std::endl;*/
    }

    // find args used by the program for fonction
    void parseProgram(int beginParser, int &beginFunction, int &endFunction,std::string name = "__OP__")
    {
        //find the fonction
        beginFunction = _program.find(name,beginParser);
        int beginArg = _program.find("(",beginFunction);
        endFunction = _program.find(")",beginFunction);


        //find args of the fonction
        int i,n;
        int j=beginArg;
        std::map<std::string,std::string>::iterator it=_map.begin();

        while(beginArg<=j && j<endFunction && it != _map.end())
        {
            //find args between '(' and ',' OR ',' and ',' OR ',' and ')' OR '(' and ')'
            i=j+1;
            j = _program.find(",",i);
            if(beginArg>j || j>=endFunction)j=endFunction;
            std::string s =_program.substr(i,j-i);
            //delete spaces characters
            while((n = s.find_first_of(" \t\n"))!=-1)s.erase(n,1);
            //insert arg in map
            it->second = s;
            it++;
        }
        endFunction++;

        /*		std::cout << "parseFunction\n";
        		// show content:
        		for ( it=_map.begin() ; it != _map.end(); it++ )
        			std::cout << (*it).first << " => " << (*it).second << std::endl;
        */
    }

    //replace function args to program args
    std::string replaceArg()
    {
        int begin = _operation.find('{')+1;
        int size = _operation.rfind('}')-begin;
        std::string function = _operation.substr(begin,size);
        std::map<std::string,std::string>::iterator it;
        for ( it=_map.begin() ; it != _map.end(); it++ )
        {
            std::string first = (*it).first;
            std::string second = (*it).second;
            int n;
            while((n=function.find(first)) != -1 && first.size()!=0) {function.replace(n,first.size(),second); std::cout << function <<"\n";}
        }

        //std::cout << "replaceArg\n" << function;
        return function;
    }

    //replace all functions by operations
    std::string replaceFunctions(std::string name = "__OP__")
    {
        int b=0,e;
        std::string s;

        parseFonction(name);

        //find all functions and replace them
        parseProgram(b,b,e,name);
        while(e>0)
        {
            s = replaceArg();
            _program.replace(b,e-b,s);
            parseProgram(b,b,e,name);

        }

        return _program;
    }

    std::string parseFile(std::string* source,std::string option,int &begin)
    {
        int endTag,startMacro,endMacro,optionPos;
        do
        {
            begin = source->find("<MACRO",begin)+1;
            endTag = source->find(">",begin);
            startMacro = source->find("_",endTag);
            endMacro = source->find("</MACRO>",endTag);
            optionPos = source->find(option,begin);

            //test error
            if(begin<=0 || endTag<0 || startMacro<0 || endMacro<0 || optionPos<0)
                return std::string();


            while(optionPos!=-1 && (source->at(optionPos-1)!= ' ' || (source->at(optionPos+option.size()) != ' ' &&  source->at(optionPos+option.size()) != '>')))
            {
                optionPos = source->find(option,optionPos+1);
            }




        }//if there is not the correct option in the MACRO also loop
        while(optionPos<begin || optionPos>endTag);
        return source->substr(startMacro,endMacro-startMacro);
    }

    std::string replaceMacros(std::string* source,std::string option)
    {
        int b=0;
        std::string name;
        bool macroExist = true;


        //find all functions and replace them
        while(macroExist)
        {

            _map.clear();
            _operation = parseFile(source,option,b);
            if(_operation.size()==0)macroExist=false;
            else
            {
                name = _operation.substr(0,_operation.find("("));
                replaceFunctions(name);
            }
        }

        return _program;
    }

};

}

}

}

#endif // SOFAOPENCL_OPENCLPROGRAMPARSER_H
