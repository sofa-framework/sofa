#ifndef PIPEPROCESS_H_
#define PIPEPROCESS_H_

#include <stdio.h>
#include <string>
#include <vector>

namespace sofa
{

namespace helper
{

namespace system
{

class PipeProcess
{
public:
    virtual ~PipeProcess();

    static bool executeProcess(const std::string &command,  const std::vector<std::string>& args, const std::string &filename, std::string & outString, std::string & errorString);

private:
    PipeProcess();

};

}
}
}
#endif /*PIPEPROCESS_H_*/
