#ifndef OPENCLPROGRAMPARSER_H
#define OPENCLPROGRAMPARSER_H

#include <iostream>
#include <vector>
#include <map>

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

    // find args used by the fonction
    void parseFonction()
    {
        //find the name of the fonction
        int begin = _operation.find("__OP__");
        begin = _operation.find("(",begin+6);
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
    void parseProgram(int beginParser, int &beginFunction, int &endFunction)
    {
        //find the fonction
        beginFunction = _program.find("__OP__",beginParser);
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
            while((n=function.find(first)) != -1)function.replace(n,first.size(),second);
        }

        //std::cout << "replaceArg\n" << function;
        return function;
    }

    //replace all functions by operations
    std::string* replaceFunctions()
    {
        int b=0,e;
        std::string s;

        parseFonction();

        //find all functions and replace them
        parseProgram(b,b,e);
        while(e!=0)
        {
            s = replaceArg();
            _program.replace(b,e-b,s);
            parseProgram(b,b,e);
            //std::cout << e;
        }

        return new std::string(_program);
    }

};

#endif // OPENCLPROGRAMPARSER_H
