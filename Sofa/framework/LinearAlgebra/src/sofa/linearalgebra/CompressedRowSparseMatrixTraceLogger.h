/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*            (c) 2006-2021 MGH, INRIA, USTL, UJF, CNRS, InSimo                *
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
#pragma once

#include <sofa/linearalgebra/config.h>

#include <cstdio>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#include <assert.h>
#include <sofa/linearalgebra/matrix_bloc_traits.h>

#ifdef SOFA_HAVE_ZLIB
#include <zlib.h>
#endif

namespace sofa::linearalgebra
{

enum FnEnum: char
{
    resizeBloc      = 0,
    compress        = 1,
    fullRows        = 3,
    setBloc         = 4,
    setBlocId       = 5,
    clearRowBloc    = 6,
    clearColBloc    = 7,
    clearRowColBloc = 8,
    clear           = 9,
    add             = 10,
    addId           = 11,
    addDBloc        = 12,
    addDValue       = 13,
    addDValueId     = 14,

    /// Specific method of CRSMatrixMechanical
    fullDiagonal    = 15,
    setVal          = 16,
    addVal          = 17,
    setValId        = 18,
    addValId        = 19,
    clearIndex      = 20,
    clearRow        = 21,
    clearCol        = 22,
    clearRowCol     = 23,

    /// Specific method of CRSMatrixConstraint
    addCol          = 24,
    setCol          = 25
};

template<typename TMatrix >
class CRSTraceWriter
{
public :
    enum { NL = TMatrix::NL };
    enum { NC = TMatrix::NC };

    typedef typename TMatrix::traits traits;
    typedef typename TMatrix::Bloc Bloc;
    typedef typename TMatrix::DBloc DBloc;
    typedef typename TMatrix::Real Real;
    typedef typename TMatrix::Policy Policy;
    typedef typename TMatrix::Index IndexType;
    CRSTraceWriter();

    CRSTraceWriter(const std::string& baseFileName, int step)
    :m_printTraceFile(nullptr)
    ,m_logTraceFile(nullptr)
    {

        std::ostringstream ss;
        ss << std::setw(4) << std::setfill('0') << step;

        std::string tbloc(traits::Name());

        m_baseFilename = baseFileName  + "_" + tbloc + "_" + ss.str();

        if constexpr(Policy::PrintTrace)
        {
            std::string printTraceFileName = m_baseFilename + ".log";
            m_printTraceFile = fopen(printTraceFileName.c_str(), "w");
        }

        if constexpr(Policy::LogTrace)
        {
            std::string logTraceFileName = m_baseFilename + ".bin";
            m_logTraceFile = fopen(logTraceFileName.c_str(), "wb" );
        }
    }



    ~CRSTraceWriter()
    {
        if(m_printTraceFile != nullptr)
        {
            fclose( m_printTraceFile );
        }

        if(m_logTraceFile != nullptr)
        {
            fclose( m_logTraceFile );
        }
    }


    template <class F, class First, class... Rest>
    void do_for(F f, First first, Rest... rest)
    {
        f(first);
        do_for(f, rest...);
    }

    template <class F>
    void do_for(F /*f*/) {}

    template <class... Args>
    void logCall_imp(const char fnId, Args... args)
    {
        logT(fnId, m_logTraceFile);

        if constexpr (Policy::PrintTrace)
        {
            std::string msg("Function ID : ");
            msg.append(std::to_string(static_cast<int>(fnId)));
            fwrite(msg.c_str(), msg.length(), 1, m_printTraceFile);
        }

        unsigned int count = 0;
        do_for([&](auto arg)
        {
            logT(arg, m_logTraceFile);
            if constexpr (Policy::PrintTrace)
            {
                std::string msg = " arg[";
                msg.append(std::to_string(count));
                msg.append("] = ");
                msg.append(ToString(arg));
                fwrite(msg.c_str(), msg.length(), 1, m_printTraceFile);
            }
            count++;
        }, args...);
        if constexpr (Policy::PrintTrace)
        {
            std::string msg = "\n";
            fwrite(msg.c_str(), msg.length(), 1, m_printTraceFile);
        }
    }

    std::string ToString (const char         c) { return std::to_string(c); }
    std::string ToString (const unsigned int i) { return std::to_string(i); }
    std::string ToString (const int          i) { return std::to_string(i); }
    std::string ToString (const double       d) { return std::to_string(d); }
    std::string ToString (const float        f) { return std::to_string(f); }

    std::string ToString (const DBloc b)
    {
        std::string bstr = "[";
        for (int i = 0; i < NL; i++)
        {
            for (int j = 0; j < NC; j++)
            {
                bstr.append(std::to_string(matrix_bloc_traits<DBloc, IndexType>::v(b, i, j)));
                if (i + j != NC + NL - 2)
                {
                    if (j == NC - 1) bstr.append(",");
                    else bstr.append(" ");
                }
            }
        }
        bstr += "]";
        return bstr;
    }

    template< class TBloc, typename std::enable_if< !std::is_same<DBloc, TBloc>::value, int >::type = 0 >
    std::string ToString (const TBloc b)
    {
        std::string bstr = "[";
        for (int i = 0; i < NL; i++)
        {
            for (int j = 0; j < NC; j++)
            {
                bstr.append(std::to_string(traits::v(b, i, j)));
                if (i + j != NC + NL - 2)
                {
                    if (j == NC - 1) bstr.append(",");
                    else bstr.append(" ");
                }
            }
        }
        bstr += "]";
        return bstr;
    }

    std::size_t logT (const char         c, FILE* file) { return fwrite( &c, sizeof(char),         1, file); }
    std::size_t logT (const unsigned int i, FILE* file) { return fwrite( &i, sizeof(unsigned int), 1, file); }
    std::size_t logT (const int          i, FILE* file) { return fwrite( &i, sizeof(int),          1, file); }
    std::size_t logT (const double       d, FILE* file) { return fwrite( &d, sizeof(double),       1, file); }
    std::size_t logT (const float        f, FILE* file) { return fwrite( &f, sizeof(float),        1, file); }

    // non-templated functions will take precedence to this method
    template< class TBloc>
    std::size_t logT (const TBloc b, FILE* file)
    {

        using traits = matrix_bloc_traits<TBloc, IndexType>;
        typename traits::Real vals[traits::NL][traits::NC];
        for (int l = 0; l < traits::NL; ++l)
            for (int c = 0; c < traits::NC; ++c)
                vals[l][c] = traits::v(b,l,c);
        return fwrite(&(vals[0][0]),sizeof(vals),1, file);
    }


    void writeMatrixToFile(const TMatrix& m)
    {
        std::string matrixFileName = m_baseFilename + ".txt";
        std::ofstream file(matrixFileName.c_str());

        m.write(file);

        file.close();
    }

protected :

    FILE* m_logTraceFile; /// File where matrix trace is logged.
    FILE* m_printTraceFile; /// File where matrix trace is logged as txt for debug purposes.

    std::string m_baseFilename;
    int         m_step;
};

template<typename Bloc, typename DBloc>
struct FnArgs
{
    int i = -1, j = -1, rowId = -1, colId = -1;
    unsigned int bi = 0, bj = 0, boffsetL = 0, boffsetC = 0;
    double v = 0;
    Bloc b = Bloc();
    DBloc bdiag = DBloc();
};

template<typename TMatrix, int matrixType >
class CRSTraceReader
{
public :
    enum { NL = TMatrix::NL };
    enum { NC = TMatrix::NC };

    typedef typename TMatrix::traits traits;
    typedef typename TMatrix::Bloc Bloc;
    typedef typename TMatrix::DBloc DBloc;
    typedef typename TMatrix::Real Real;
    typedef typename TMatrix::Policy Policy;
    typedef typename TMatrix::Index IndexType;

    CRSTraceReader() = default;

    CRSTraceReader(const std::vector<int>& fnIds, const std::vector<FnArgs<typename TMatrix::Bloc, typename TMatrix::DBloc>>& fnArgs)
    :m_fnIds(fnIds)
    ,m_fnArgs(fnArgs)
    {

    }

    const std::vector<int>& getFnIds() const
    {
        return m_fnIds;
    }

    const std::vector<FnArgs<typename TMatrix::Bloc, typename TMatrix::DBloc>>& getFnArgs() const
    {
        return m_fnArgs;
    }

    void readInstructions(const std::string& logTraceFileName)
    {
#ifdef SOFA_HAVE_ZLIB
        if (logTraceFileName.substr(logTraceFileName.length() - 3) == ".gz")
        {
            auto currentFile = fileOpenGZ(logTraceFileName.c_str(), "rb");
            if (currentFile == nullptr)
            {
                std::cout << "ERROR: when reading trace instructions from file : " << logTraceFileName << std::endl;
            }
            setFileGZ(currentFile);
        }
        else
#endif
        {
            auto currentFile = fileOpen(logTraceFileName.c_str(), "rb");
            if (currentFile == nullptr)
            {
                std::cout << "ERROR: when reading trace instructions from file : " << logTraceFileName << std::endl;
            }
            setFile(currentFile);
        }

        int fnId = -1;
        while (readFn(fnId))
        {
            m_fnIds.push_back(fnId);
            m_fnArgs.push_back(readArgs(m_fnIds.back()));
        }

        fileClose();
    }

    void playInstructions(TMatrix& m)
    {
        for (std::size_t j = 0; j < m_fnIds.size(); ++j)
        {
            callFn(m, m_fnIds[j], m_fnArgs[j]);
        }
    }

    void openDebugLogFile(const std::string& fileName) 
    { 
        m_debugLogFile.open(fileName); 
    }

    void closeDebugLogFile()
    {
        if (m_debugLogFile.is_open()) m_debugLogFile.close();
    }

private:

    std::size_t readT(char& c, int argId = -1)
    {
        std::size_t processed = this->fileRead( &c, sizeof(char), 1);
        if(m_debugLogFile.is_open())
        {
            if (argId != -1 && processed != 0) m_debugLogFile << " arg[" << argId << "] = " << c;
        }
        return processed;
    }
    std::size_t readT(unsigned int& i, int argId = -1)
    {
        std::size_t processed = this->fileRead( &i, sizeof(unsigned int), 1);
        if (m_debugLogFile.is_open())
        {
            if (argId != -1 && processed != 0) m_debugLogFile << " arg[" << argId << + "] = " << i;
        }
        return processed;
    }
    std::size_t readT(int& i, int argId = -1)
    {
        std::size_t processed = this->fileRead( &i, sizeof(int), 1);
        if (m_debugLogFile.is_open())
        {
            if (argId != -1 && processed != 0) m_debugLogFile << " arg[" << argId << "] = " << i;
        }
        return processed;
    }

    // non-templated functions will take precedence over this method
    template< class TBloc>
    std::size_t readT(TBloc& b, int argId = -1)
    {
        using traits = matrix_bloc_traits<TBloc, IndexType>;
        typename traits::Real vals[traits::NL][traits::NC] { { 0 } };
        std::size_t processed = fileRead(&(vals[0][0]),sizeof(vals),1);
        for (int l = 0; l < traits::NL; ++l)
            for (int c = 0; c < traits::NC; ++c)
                traits::vset(b,l,c,vals[l][c]);
        if (m_debugLogFile.is_open())
        {
            if (argId != -1 && processed != 0) m_debugLogFile << " arg[" << argId << "] = " << b;
        }
        return processed;
    }

    std::size_t readFn(int& fnId)
    {
        char tempFnId;
        std::size_t processed = readT(tempFnId);
        fnId = static_cast<int>(tempFnId);
        if (m_debugLogFile.is_open())
        {
            if (processed != 0) m_debugLogFile << "Function ID : " << fnId;
        }
        return processed;
    }

    FnArgs<Bloc, DBloc> readArgs(int fnId)
    {
        FnArgs<Bloc, DBloc> args;
        switch (fnId)
        {
            case FnEnum::resizeBloc      : readT(args.i, 0); readT(args.j, 1); break;
            case FnEnum::compress        : break;
            case FnEnum::setBloc         : readT(args.i, 0); readT(args.j, 1); readT(args.b, 2); break;
            case FnEnum::setBlocId       : readT(args.i, 0); readT(args.j, 1); readT(args.rowId, 2); readT(args.colId, 3); readT(args.b, 4); break;
            case FnEnum::clearRowBloc    : readT(args.i, 0); break;
            case FnEnum::clearColBloc    : readT(args.i, 0); break;
            case FnEnum::clearRowColBloc : readT(args.i, 0); break;
            case FnEnum::clear           : break;
            case FnEnum::add             : readT(args.bi, 0); readT(args.bj, 1); readT(args.b, 2); break;
            case FnEnum::addId           : readT(args.bi, 0); readT(args.bj, 1); readT(args.rowId, 2); readT(args.colId, 3); readT(args.b, 4); break;
            case FnEnum::addDBloc        : readT(args.bi, 0); readT(args.bj, 1); readT(args.bdiag, 2); break;
            case FnEnum::addDValue       : readT(args.bi, 0); readT(args.bj, 1); readT(args.v, 2);  break;
            case FnEnum::addDValueId     : readT(args.bi, 0); readT(args.bj, 1); readT(args.rowId, 2); readT(args.colId, 3); readT(args.v, 4); break;
            case FnEnum::fullRows        : break;
            case FnEnum::fullDiagonal    : break;
            case FnEnum::setVal          : readT(args.i, 0); readT(args.j, 1); readT(args.v, 2); break;
            case FnEnum::addVal          : readT(args.i, 0); readT(args.j, 1); readT(args.v, 2); break;
            case FnEnum::setValId        : readT(args.i, 0); readT(args.j, 1); readT(args.rowId, 2); readT(args.colId, 3); readT(args.v, 4); break;
            case FnEnum::addValId        : readT(args.i, 0); readT(args.j, 1); readT(args.rowId, 2); readT(args.colId, 3); readT(args.v, 4); break;
            case FnEnum::clearIndex      : readT(args.i, 0); readT(args.j, 1); break;
            case FnEnum::clearRow        : readT(args.i, 0); break;
            case FnEnum::clearCol        : readT(args.i, 0); break;
            case FnEnum::clearRowCol     : readT(args.i, 0); break;
            case FnEnum::addCol          : readT(args.i, 0); readT(args.j, 1); readT(args.b, 2); break;
            case FnEnum::setCol          : readT(args.i, 0); readT(args.j, 1); readT(args.b, 2); break;
            default                      : 
            {
                std::cout << "Unrecognized function id " << fnId << std::endl;
                assert(false);
                break;
            }
        }
        if (m_debugLogFile.is_open()) m_debugLogFile << std::endl;
        return args;
    }


    void defaultcallFn(TMatrix& m, int fnId, FnArgs<Bloc, DBloc>& args)
    {
        switch (fnId)
        {
            case FnEnum::resizeBloc      : m.resizeBloc(args.i, args.j); break;
            case FnEnum::compress        : m.compress(); break;
            case FnEnum::setBloc         : m.setBloc(args.i, args.j, args.b); break;
            case FnEnum::setBlocId       : m.setBloc(args.i, args.j, args.rowId, args.colId, args.b); break;
            case FnEnum::clearRowBloc    : m.clearRowBloc(args.i); break;
            case FnEnum::clearColBloc    : m.clearColBloc(args.i); break;
            case FnEnum::clearRowColBloc : m.clearRowColBloc(args.i); break;
            case FnEnum::clear           : m.clear(); break;
            case FnEnum::fullRows        : m.fullRows(); break;
            default: 
            {
                std::cout << "Unrecognized function id " << fnId << std::endl;
                assert(false);
                return;
            }
        }
    }

    template <int Type = matrixType>
    typename std::enable_if< Type == 0 >::type
    callFn(TMatrix& m, int fnId, FnArgs<Bloc, DBloc>& args)
    {
        return defaultcallFn(m, fnId, args);
    }

    template <int Type = matrixType>
    typename std::enable_if< Type == 1 >::type
    callFn(TMatrix& m, int fnId, FnArgs<Bloc, DBloc>& args)
    {
        switch (fnId)
        {
            case FnEnum::add             : m.add(args.bi, args.bj, args.b); break;
            case FnEnum::addId           : m.add(args.bi, args.bj, args.rowId, args.colId, args.b); break;
            case FnEnum::addDBloc        : m.addDBloc(args.bi, args.bj, args.bdiag); break;
            case FnEnum::addDValue       : m.addDValue(args.bi, args.bj, args.v); break;
            case FnEnum::addDValueId     : m.addDValue(args.bi, args.bj, args.rowId, args.colId, args.v); break;

            case FnEnum::fullDiagonal    : m.fullDiagonal();  break;
            case FnEnum::setVal          : m.set(args.i, args.j, args.v); break;
            case FnEnum::addVal          : m.add(args.i, args.j, args.v); break;
            case FnEnum::setValId        : m.set(args.i, args.j, args.rowId, args.colId, args.v); break;
            case FnEnum::addValId        : m.add(args.i, args.j, args.rowId, args.colId, args.v); break;
            case FnEnum::clearIndex      : m.clear(args.i, args.j); break;
            case FnEnum::clearRow        : m.clearRow(args.i); break;
            case FnEnum::clearCol        : m.clearCol(args.i); break;
            case FnEnum::clearRowCol     : m.clearRowCol(args.i); break;
            default: defaultcallFn(m, fnId, args); return;
        }
    }

    template <int Type = matrixType>
    typename std::enable_if< Type == 2 >::type
    callFn(TMatrix& m, int fnId, FnArgs<Bloc, DBloc>& args)
    {
        switch (fnId)
        {
            case FnEnum::addCol          : m.writeLine(args.i).addCol(args.j, args.b); break;
            case FnEnum::setCol          : m.writeLine(args.i).setCol(args.j, args.b); break;
            default : defaultcallFn(m, fnId, args); return;
        }
    }

    static FILE* fileOpen(const char *path, const char *mode)
    {
        return fopen(path, mode);
    }
    
    void setFile(FILE* file)
    {
        m_matrixFile = file;
#ifdef SOFA_HAVE_ZLIB
        m_matrixFileGZ = nullptr;
#endif
    }

#ifdef SOFA_HAVE_ZLIB
    static gzFile fileOpenGZ(const char *path, const char *mode)
    {
        return gzopen(path, mode);
    }
    void setFileGZ(gzFile file)
    {
        m_matrixFile = nullptr;
        m_matrixFileGZ = file;
    }
#endif

    std::size_t fileRead(void* buf, std::size_t size, std::size_t nitems)
    {
#ifdef SOFA_HAVE_ZLIB
        if (m_matrixFileGZ)
        {
#if ZLIB_VERNUM >= 0x1290
            return gzfread(buf, size, nitems, m_matrixFileGZ);
#else
            return gzread(m_matrixFileGZ, buf, (unsigned)nitems*size)/size;
#endif
        }
#endif
        if (m_matrixFile)
        {
            return fread(buf, size, nitems, m_matrixFile);
        }

        return 0;
    }
    void fileClose()
    {
#ifdef SOFA_HAVE_ZLIB
        if (m_matrixFileGZ)
        {
            gzclose(m_matrixFileGZ);
            m_matrixFileGZ = nullptr;
        }
#endif
        if (m_matrixFile)
        {
            fclose(m_matrixFile);
            m_matrixFile = nullptr;
        }
    }

    FILE* m_matrixFile = nullptr; /// File where matrix trace is logged.
#ifdef SOFA_HAVE_ZLIB
    gzFile m_matrixFileGZ = nullptr; /// File where matrix trace is logged.
#endif
    std::ofstream m_debugLogFile; /// File where matrix trace is logged as txt for debug purposes.
    TMatrix * m_CRSmatrix;

    std::vector<int> m_fnIds;
    std::vector<FnArgs<typename TMatrix::Bloc, typename TMatrix::DBloc>> m_fnArgs;
};

} // namespace sofa::linearalgebra
