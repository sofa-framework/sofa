#include "PipeProcess.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <cstring>
#include <iostream>
#include <sstream>

#define BUFSIZE (64*1024-1)
#define STEPSIZE (1024)
//#define STEPSIZE BUFSIZE

namespace sofa
{

namespace helper
{

namespace system
{

ssize_t writeall(int fd, const void* buf, size_t count)
{
    size_t total = 0;
    while (count > total)
    {
        ssize_t r = write(fd, ((const char*)buf)+total, count-total);
        if (r < 0) return r;
        total += r;
    }
    return total;
}

ssize_t writecheck(int fd, const void* buf, size_t count)
{
    if (count > 5 && ((const char*)buf)[1]=='<' && ((const char*)buf)[2]=='<' && ((const char*)buf)[3]=='>' && ((const char*)buf)[4]=='>')
    {
        char space=' ';
        writeall(fd,&space,1);
        return writeall(fd,((const char*)buf)+5,count-5);
    }
    else if (count > 6 && ((const char*)buf)[1]=='<' && ((const char*)buf)[2]=='<' && ((const char*)buf)[4]=='>' && ((const char*)buf)[5]=='>')
    {
        writeall(fd,((const char*)buf)+3,1);
        return writeall(fd,((const char*)buf)+6,count-6);
    }
    else
        return writeall(fd,buf,count);
}


PipeProcess::PipeProcess()
{
}

PipeProcess::~PipeProcess()
{
}

bool PipeProcess::executeProcess(const std::string &command,  const std::vector<std::string> &args, const std::string &filename, std::string & outString, std::string & errorString)
{
    int fds[2][2];
    pid_t   pid;

    //char eol = '\n';
    char** cargs;
    cargs = new char* [args.size()+2];
    cargs[0] = (char*)command.c_str();
    for (unsigned int i=1 ; i< args.size() + 1 ; i++)
        cargs[i] = (char*)args[i].c_str();
    cargs[args.size() + 1] = NULL;



    int filefd = open(filename.c_str(),O_RDONLY);

    int fdin = 0;
    int fdout = 1;
    outString = "";
    errorString = "";
    std::stringstream outStream;
    std::stringstream errorStream;

    if (pipe(fds[0]) || pipe(fds[1]))
    {
        std::cerr << "pipe failed."<<std::endl;
        return false;
    }
    pid = fork();
    if (pid < 0)
    {
        std::cerr << "fork failed."<<std::endl;
        return false;
    }
    else if (pid == 0)
    {
        // child process
        close(fds[0][0]);
        close(fds[1][0]);
        // Remove standard input
        //dup2(open("/dev/null",O_RDONLY),0);
        dup2(filefd, 0);
        dup2(fds[0][1],1);
        dup2(fds[1][1],2);

        //int retexec = execlp("/bin/sh","/bin/sh","-c", command.c_str() ,NULL);
        int retexec = execvp(command.c_str(), cargs);
        //int retexec = execlp(command.c_str(), command.c_str(), NULL);
        //int retexec = execlp("wc", "wc", NULL);
        std::cerr << "PipeProcess : ERROR: execlp( "<< command.c_str() << " " ;
        for (unsigned int i=0; i<args.size() + 1 ; i++)
            std::cerr << cargs[i] << " ";
        std::cerr << ") returned "<<retexec<<std::endl;
        return false;
    }
    else
    {
        // parent process
        char inbuf[BUFSIZE];
        char buf[2][BUFSIZE];
        int nfill[2];
        close(fds[0][1]);
        close(fds[1][1]);
        fd_set rfds;
        int nfd = fdin+1;
        FD_ZERO(&rfds);
        FD_SET(fdin, &rfds);
        int nopen = 0;
        for (int i=0; i<2; i++)
        {
            //fcntl(fds[i][0],F_SETFL, fcntl(fds[i][0],F_GETFL)|O_NONBLOCK );
            FD_SET(fds[i][0], &rfds);
            if (fds[i][0] >= nfd) nfd = fds[i][0]+1;
            // add prefixes
            //buf[i][0] = '0'+i;
            //nfill[i] = 1;
            nfill[i] = 0;
            ++nopen;
        }
        fd_set ready;
        ready = rfds;
        while (nopen> 0 && select(nfd, &ready, NULL, NULL, NULL)> 0)
        {
            if (FD_ISSET(fdin, &ready))
            {
                int n = read(fdin, inbuf, BUFSIZE);
                if (n>0)
                {
                    writeall(2,inbuf,n);
                }
                else if (n==0)
                {
                    FD_CLR(fdin, &rfds);
                }
            }
            for (int i=0; i<2; i++)
            {
                if (FD_ISSET(fds[i][0], &ready))
                {
                    int n = BUFSIZE-nfill[i];
                    if (n> STEPSIZE) n = STEPSIZE;
                    n = read(fds[i][0], buf[i]+nfill[i], n);
                    if (n==0)
                    {
                        if (nfill[i]> 1)
                        {
                            buf[i][nfill[i]] = '\n';
                            //writecheck(fdout,buf[i],nfill[i]+1);
                            if (i==0)
                                outStream << std::string(buf[i],nfill[i]+1);
                            else
                                errorStream << std::string(buf[i],nfill[i]+1);
                        }
                        --nopen;
                        FD_CLR(fds[i][0], &rfds);
                    }
                    else if (n> 0)
                    {
                        int start = 0;
                        while (n>0)
                        {
                            while (n> 0 && buf[i][nfill[i]] != '\n' && buf[i][nfill[i]] != '\r')
                            {
                                ++nfill[i];
                                --n;
                            }
                            if (n> 0 && buf[i][nfill[i]] == '\n')
                            {
                                // write line
                                //writecheck(fdout,buf[i]+start,nfill[i]+1-start);
                                if (i==0)
                                    outStream << std::string(buf[i]+start,nfill[i]-start);
                                else
                                    errorStream << std::string(buf[i]+start,nfill[i]-start);
                                if (n> 0 && nfill[i] < BUFSIZE && buf[i][nfill[i]+1] == '\r')
                                {
                                    // ignore '\r' after '\n'
                                    ++nfill[i];
                                    --n;
                                }
                                start = nfill[i];
                                //buf[i][nfill[i]] = '0'+i;
                                ++nfill[i];
                                --n;
                            }
                            else if (n> 0 && buf[i][nfill[i]] == '\r')
                            {
                                // replace with '\n'								//////buf[i][nfill[i]] = '\n';
                                // write line
                                //writecheck(fdout,buf[i]+start,nfill[i]+1-start);
                                if (i == 0)
                                    outStream << std::string(buf[i]+start,nfill[i]-start);
                                else
                                    errorStream << std::string(buf[i]+start,nfill[i]-start);
                                if (n> 0 && nfill[i] < BUFSIZE && buf[i][nfill[i]+1] == '\n')
                                {
                                    // ignore '\n' after '\r'
                                    ++nfill[i];
                                    --n;
                                }
                                start = nfill[i];
                                //buf[i][nfill[i]] = '0'+i;
                                ++nfill[i];
                                --n;
                            }
                        }
                        if (start> 0)
                        {
                            for (int j=start; j<nfill[i]; j++)
                                buf[i][j-start] = buf[i][j];
                            nfill[i] -= start;
                        }
                        if (nfill[i] == BUFSIZE)
                        {
                            // line too long -> split
                            //writecheck(fdout,buf[i],nfill[i]);
                            //writecheck(fdout,&eol,1);
                            if (i == 0)
                                outStream << std::string(buf[i],nfill[i]) << std::endl;
                            else
                                errorStream << std::string(buf[i],nfill[i]) << std::endl;
                            //buf[i][0] = '0'+i;
                            nfill[i] = 1;
                        }
                    }
                }
            }
            ready = rfds;
        }
        close(fds[0][0]);
        close(fds[1][0]);
        int status=0;
        waitpid(pid,&status,0);

//		const char* msg;
//		if (status == 0) msg = " OK\n";
//		else msg = " Failed\n";
//		writeall(fdout,msg,strlen(msg));

        if (fdout != 1)
            close(fdout);
        close(filefd);

        outString = outStream.str();
        errorString = errorStream.str();
        return (status == 0);
    }
}

}
}
}
