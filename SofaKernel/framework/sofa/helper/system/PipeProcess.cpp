/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include "PipeProcess.h"

#ifdef WIN32
#include <windows.h>
#include <process.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
typedef int ssize_t;
typedef HANDLE fd_t;
typedef SOCKET socket_t;
#elif defined(_XBOX)
#include <xtl.h>
#elif defined(PS3)
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
typedef int fd_t;
typedef int socket_t;
#endif

#include <cstring>
#include <iostream>
#include <sstream>

#define BUFSIZE (64*1024-1)
#define STEPSIZE (1024)
//#define STEPSIZE BUFSIZE

#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace helper
{

namespace system
{

PipeProcess::PipeProcess()
{
}

PipeProcess::~PipeProcess()
{
}
/**   File as Stdin for windows does not work (yet)
  *   So the filename must be given into the args vector
  *   and argument filenameStdin is currently ignored
  */

bool PipeProcess::executeProcess(const std::string &command,  const std::vector<std::string> &args, const std::string &/*filenameStdin*/, std::string & outString, std::string & errorString)
{
#if defined (_XBOX) || defined(PS3)
    return false; // not supported
#else
    //std::string fileIN = filenameStdin;
    //Remove this line below and uncomment the one above when Windows will be able to read file as stdin
    std::string fileIN = "";

    fd_t fds[2][2];

    //char eol = '\n';
    char** cargs;

    cargs = new char* [args.size()+2];
    cargs[0] = (char*)command.c_str();
    for (unsigned int i=1 ; i< args.size() + 1 ; ++i)
        cargs[i] = (char*)args[i-1].c_str();
    cargs[args.size() + 1] = NULL;

    fd_t fdin;
//    fd_t fdout;
//    fd_t fderr;

#ifdef WIN32
    fdin = GetStdHandle(STD_INPUT_HANDLE);

//    fdout = GetStdHandle(STD_OUTPUT_HANDLE);
//    fderr = GetStdHandle(STD_ERROR_HANDLE);

    std::string newCommand(command);
    for (unsigned int i=0 ; i< args.size() ; ++i)
        newCommand += " " + args[i];

#else
    fdin = 0;
//    fdout = 1;
//    fderr = 2;
#endif

    outString = "";
    errorString = "";
    std::stringstream outStream;
    std::stringstream errorStream;

#ifdef WIN32
    SECURITY_ATTRIBUTES saAttr;
    // Set the bInheritHandle flag so pipe handles are inherited.
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = NULL;
    if (!CreatePipe(&fds[0][0],&fds[0][1],&saAttr,0) || !CreatePipe(&fds[1][0],&fds[1][1],&saAttr,0))
#else
    if (pipe(fds[0]) || pipe(fds[1]))
#endif
    {
        msg_error("PipeProcess") << "pipe failed.";
        delete [] cargs;
        return false;
    }
#ifdef WIN32
    HANDLE hFile = 0;
    if (fileIN != "")
        hFile = CreateFileA( (fileIN.c_str()),
                GENERIC_READ,
                FILE_SHARE_READ,
                NULL,
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                NULL);

    if (hFile == INVALID_HANDLE_VALUE)
    {
        msg_error("PipeProcess") << "failed to open file for stdin";
    }

    // Ensure that the read handle to the child process's pipe for STDOUT is not inherited.
    SetHandleInformation( fds[0][0], HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation( fds[1][0], HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation( fdin, HANDLE_FLAG_INHERIT, 1);
    PROCESS_INFORMATION piProcInfo;
    STARTUPINFOA siStartInfo;
    ZeroMemory( &piProcInfo, sizeof(PROCESS_INFORMATION) );
    ZeroMemory( &siStartInfo, sizeof(STARTUPINFOA) );
    siStartInfo.cb = sizeof(STARTUPINFOA);
    if (fileIN != "")
        siStartInfo.hStdInput = hFile;
    else siStartInfo.hStdInput = fdin;

    siStartInfo.hStdOutput = fds[0][1];
    siStartInfo.hStdError = fds[1][1];
    siStartInfo.dwFlags |= STARTF_USESTDHANDLES;
    if (!CreateProcessA(NULL,
            (char*)newCommand.c_str(),       // command line
            NULL,          // process security attributes
            NULL,          // primary thread security attributes
            TRUE,          // handles are inherited
            0,             // creation flags
            NULL,          // use parent's environment
            NULL,          // use parent's current directory
            &siStartInfo,  // STARTUPINFO pointer
            &piProcInfo))  // receives PROCESS_INFORMATION
    {
        msg_error("PipeProcess") << "CreateProcess failed : "<<GetLastError();
        delete [] cargs;
        return 1;
    }

    {
        // parent process
        //char inbuf[BUFSIZE];
        char buf[2][BUFSIZE];
        int nfill[2];
        CloseHandle(fds[0][1]);
        CloseHandle(fds[1][1]);
        unsigned long exit = 0;
        for (int i=0; i<2; ++i)
            nfill[i] = 0;
        for(int i=0;; ++i)
        {
            GetExitCodeProcess(piProcInfo.hProcess,&exit);      //while the process is running
            if (exit != STILL_ACTIVE)
                break;

            bool busy = false;
            for (int i=0; i<2; ++i)
            {
                DWORD n = BUFSIZE-nfill[i];
                if (n > STEPSIZE) n = STEPSIZE;
                DWORD bread = 0;
                DWORD avail = 0;
                PeekNamedPipe(fds[i][0],buf[i]+nfill[i],n,&bread,&avail,NULL);

                if (bread>0)
                {
                    busy = true;
                    ReadFile(fds[i][0], buf[i]+nfill[i], n, &n, NULL);
                    nfill[i] += n;
                    {
                        // write line
                        if (i==0)
                            outStream << std::string(buf[i],nfill[i]);
                        else
                            errorStream << std::string(buf[i],nfill[i]);

                        nfill[i] = 0;
                    }
                }
            }
            if (!busy)
                Sleep(0);
        }
        CloseHandle(fds[0][0]);
        CloseHandle(fds[1][0]);
        int status=exit;
        CloseHandle(piProcInfo.hProcess);
        CloseHandle(piProcInfo.hThread);
        //waitpid(pid,&status,0);
#else
    int filefd = 0;
    if (fileIN != "")
        filefd = open(fileIN.c_str(),O_RDONLY);

    pid_t   pid;
    pid = fork();
    if (pid < 0)
    {
        msg_error("PipeProcess") << "fork failed.";
        delete [] cargs;
        return false;
    }
    else if (pid == 0)
    {
        // child process
        close(fds[0][0]);
        close(fds[1][0]);
        // Remove standard input
        if (fileIN != "")
            dup2(filefd, 0);
        else dup2(open("/dev/null",O_RDONLY),0);

        dup2(fds[0][1],1);
        dup2(fds[1][1],2);

        int retexec = execvp(command.c_str(), cargs);
        helper::logging::MessageDispatcher::LoggerStream msgerror = msg_error("PipeProcess");
        msgerror << "execlp( "<< command.c_str() << " " ;
        for (unsigned int i=0; i<args.size() + 1 ; ++i)
            msgerror << cargs[i] << " ";
        msgerror << ") returned "<<retexec;
        delete [] cargs;
        return false;
    }
    else
    {
        // parent process
//		char inbuf[BUFSIZE];
        char buf[2][BUFSIZE];
        int nfill[2];
        close(fds[0][1]);
        close(fds[1][1]);
        fd_set rfds;
        int nfd = fdin+1;
        FD_ZERO(&rfds);
        FD_SET(fdin, &rfds);
        int nopen = 0;
        for (int i=0; i<2; ++i)
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
            //read stdin
//			if (FD_ISSET(fdin, &ready))
//			{
//				int n = read(fdin, inbuf, BUFSIZE);
//				if (n>0)
//				{
//					writeall(2,inbuf,n);
//				}
//				else if (n==0)
//				{
//					FD_CLR(fdin, &rfds);
//				}
//			}
            for (int i=0; i<2; ++i)
            {
                if (FD_ISSET(fds[i][0], &ready))
                {
                    int n = BUFSIZE-nfill[i];
                    if (n> STEPSIZE) n = STEPSIZE;
                    n = read(fds[i][0], buf[i]+nfill[i], n);

                    if (n == 0)
                    {
                        if (nfill[i]> 1)
                        {
                            buf[i][nfill[i]] = '\n';
                            if (i==0)
                                outStream << std::string(buf[i],nfill[i]+1);
                            else
                                errorStream << std::string(buf[i],nfill[i]+1);
                        }
                        --nopen;
                        FD_CLR(fds[i][0], &rfds);
                    }
                    else
                    {
                        nfill[i] += n;

                        // write line
                        if (i==0)
                            outStream << std::string(buf[i],nfill[i]);
                        else
                            errorStream << std::string(buf[i],nfill[i]);

                        nfill[i] = 0;
                    }

                }
            }
            ready = rfds;
        }
        close(fds[0][0]);
        close(fds[1][0]);
        int status=0;
        waitpid(pid,&status,0);

//        if (fdout != 1)
//            close(fdout);
        close(filefd);
#endif


        outString = outStream.str();
        errorString = errorStream.str();
        delete [] cargs;
        return (status == 0);
    }
#endif
}

}
}
}

