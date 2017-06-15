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
#define SOFA_HELPER_ADVANCEDTIMER_CPP
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/system/thread/thread_specific_ptr.h>
#include <sofa/helper/system/atomic.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/map.h>

#include <cmath>
#include <cstdlib>
#include <stack>

#define DEFAULT_INTERVAL 100


namespace sofa
{

namespace helper
{
#if defined(_XBOX) || defined(__PS3__)
char* getenv(const char* varname) { return NULL; } // NOT IMPLEMENTED
#endif

typedef sofa::helper::system::thread::ctime_t ctime_t;
typedef sofa::helper::system::thread::CTime CTime;

template class SOFA_HELPER_API AdvancedTimer::Id<AdvancedTimer::Timer>;
template class SOFA_HELPER_API AdvancedTimer::Id<AdvancedTimer::Step>;
template class SOFA_HELPER_API AdvancedTimer::Id<AdvancedTimer::Obj>;
template class SOFA_HELPER_API AdvancedTimer::Id<AdvancedTimer::Val>;

class Record
{
public:
    ctime_t time;
    enum Type { RNONE, RBEGIN, REND, RSTEP_BEGIN, RSTEP_END, RSTEP, RVAL_SET, RVAL_ADD } type;
    unsigned int id;
    unsigned int obj;
    double val;
    Record() : type(RNONE), id(0), obj(0), val(0) {}
};

class TimerData
{
public:
    AdvancedTimer::IdTimer id;
    helper::vector<Record> records;
    int nbIter;
    int interval;
    int defaultInterval;

    class StepData
    {
    public:
        int level;
        int num, numIt;
        ctime_t tstart;
        ctime_t tmin;
        ctime_t tmax;
        ctime_t ttotal;
        ctime_t ttotal2;
        int lastIt;
        ctime_t lastTime;
        StepData() : level(0), num(0), numIt(0), tstart(0), tmin(0), tmax(0), ttotal(0), ttotal2(0), lastIt(-1), lastTime(0) {}
    };

    std::map<AdvancedTimer::IdStep, StepData> stepData;
    helper::vector<AdvancedTimer::IdStep> steps;

    class ValData
    {
    public:
        int num, numIt;
        double vmin;
        double vmax;
        double vtotal;
        double vtotal2;
        double vtotalIt;
        int lastIt;
        ValData() : num(0), numIt(0), vmin(0), vmax(0), vtotal(0), vtotal2(0), vtotalIt(0), lastIt(-1) {}
    };

    std::map<AdvancedTimer::IdVal, ValData> valData;
    helper::vector<AdvancedTimer::IdVal> vals;

    TimerData()
        : nbIter(-1), interval(0), defaultInterval(DEFAULT_INTERVAL)
    {
    }

    void init(AdvancedTimer::IdTimer id)
    {
        this->id = id;
        std::string envvar = std::string("SOFA_TIMER_") + (std::string)id;
        const char* val = getenv(envvar.c_str());
        if (!val || !*val)
            val = getenv("SOFA_TIMER_ALL");
        if (val && *val)
            interval = atoi(val);
        else
            interval = 0;
        defaultInterval = (interval != 0) ? interval : DEFAULT_INTERVAL;
    }
    void clear();
    void process();
    void print();
    void print(std::ostream& result);
};

std::map< AdvancedTimer::IdTimer, TimerData > timers;

helper::system::atomic<int> activeTimers;
SOFA_THREAD_SPECIFIC_PTR(std::stack<AdvancedTimer::IdTimer>, curTimerThread);
SOFA_THREAD_SPECIFIC_PTR(helper::vector<Record>, curRecordsThread);

std::stack<AdvancedTimer::IdTimer>& getCurTimer()
{
    std::stack<AdvancedTimer::IdTimer>* ptr = curTimerThread;
    if (!ptr)
    {
        ptr = new std::stack<AdvancedTimer::IdTimer>;
        curTimerThread = ptr;
    }
    return *ptr;
}

helper::vector<Record>* getCurRecords()
{
    if (!activeTimers) return NULL;
    return curRecordsThread;
}

void setCurRecords(helper::vector<Record>* ptr)
{
    helper::vector<Record>* prev = curRecordsThread;
    curRecordsThread = ptr;
    if (ptr && !prev) ++activeTimers;
    else if (!ptr && prev) --activeTimers;
}

AdvancedTimer::SyncCallBack syncCallBack = NULL;
void* syncCallBackData = NULL;

std::pair<AdvancedTimer::SyncCallBack,void*> AdvancedTimer::setSyncCallBack(SyncCallBack cb, void* userData)
{
    std::pair<AdvancedTimer::SyncCallBack,void*> old;
    old.first = syncCallBack;
    old.second = syncCallBackData;
    syncCallBack = cb;
    syncCallBackData = userData;
    return old;
}

void AdvancedTimer::clear()
{
    setCurRecords(NULL);
    std::stack<AdvancedTimer::IdTimer>* ptr = curTimerThread;
    if (ptr)
        while (!ptr->empty())
            ptr->pop();
    if (activeTimers == 0)
        timers.clear();
}

bool AdvancedTimer::isEnabled(IdTimer id)
{
    TimerData& data = timers[id];
    if (!data.id)
    {
        data.init(id);
    }
    return (data.interval != 0);
}

void AdvancedTimer::setEnabled(IdTimer id, bool val)
{
    TimerData& data = timers[id];
    if (!data.id)
    {
        data.init(id);
    }
    if (val && data.interval == 0)
        data.interval = data.defaultInterval;
    else if (!val && data.interval != 0)
        data.interval = 0;
}

int  AdvancedTimer::getInterval(IdTimer id)
{
    TimerData& data = timers[id];
    if (!data.id)
    {
        data.init(id);
    }
    return (data.interval ? data.interval : data.defaultInterval);
}

void AdvancedTimer::setInterval(IdTimer id, int val)
{
    TimerData& data = timers[id];
    if (!data.id)
    {
        data.init(id);
    }
    data.defaultInterval = val;
    if (data.interval) data.interval = val;
}

void AdvancedTimer::begin(IdTimer id)
{
    std::stack<AdvancedTimer::IdTimer>& curTimer = getCurTimer();
    curTimer.push(id);
    TimerData& data = timers[curTimer.top()];
    if (!data.id)
    {
        data.init(id);
    }
    if (data.interval == 0)
    {
        setCurRecords(NULL);
        return;
    }
    helper::vector<Record>* curRecords = &(data.records);
    setCurRecords(curRecords);
    curRecords->clear();
    if (syncCallBack) (*syncCallBack)(syncCallBackData);
    Record r;
    r.time = CTime::getTime();
    r.type = Record::RBEGIN;
    r.id = id;
    curRecords->push_back(r);
}

void AdvancedTimer::end(IdTimer id, std::ostream& result)
{
    std::stack<AdvancedTimer::IdTimer>& curTimer = getCurTimer();

    if (curTimer.empty())
    {
        msg_error("AdvancedTimer::end") << "timer[" << id << "] called while begin was not" ;
        return;
    }
    if (id != curTimer.top())
    {
        msg_error("AdvancedTimer::end") << "timer[" << id << "] does not correspond to last call to begin(" << curTimer.top() << ")" ;
        return;
    }
    helper::vector<Record>* curRecords = getCurRecords();
    if (curRecords)
    {
        if (syncCallBack) (*syncCallBack)(syncCallBackData);
        Record r;
        r.time = CTime::getTime();
        r.type = Record::REND;
        r.id = id;
        curRecords->push_back(r);

        TimerData& data = timers[curTimer.top()];
        data.process();
        if (data.nbIter == data.interval)
        {
            data.print(result);
            data.clear();
        }
    }
    curTimer.pop();
    if (curTimer.empty())
    {
        setCurRecords(NULL);
    }
    else
    {
        TimerData& data = timers[curTimer.top()];
        setCurRecords((data.interval == 0) ? NULL : &(data.records));
    }
}

void AdvancedTimer::end(IdTimer id)
{
    std::stack<AdvancedTimer::IdTimer>& curTimer = getCurTimer();

    if (curTimer.empty())
    {
        msg_error("AdvancedTimer::end") << "timer[" << id << "] called while begin was not" ;
        return;
    }
    if (id != curTimer.top())
    {
        msg_error("AdvancedTimer::end") << "timer[" << id << "] does not correspond to last call to begin(" << curTimer.top() << ")" ;
        return;
    }
    helper::vector<Record>* curRecords = getCurRecords();
    if (curRecords)
    {
        if (syncCallBack) (*syncCallBack)(syncCallBackData);
        Record r;
        r.time = CTime::getTime();
        r.type = Record::REND;
        r.id = id;
        curRecords->push_back(r);

        TimerData& data = timers[curTimer.top()];
        data.process();
        if (data.nbIter == data.interval)
        {
            data.print();
            data.clear();
        }
    }
    curTimer.pop();
    if (curTimer.empty())
    {
        setCurRecords(NULL);
    }
    else
    {
        TimerData& data = timers[curTimer.top()];
        setCurRecords((data.interval == 0) ? NULL : &(data.records));
    }
}

bool AdvancedTimer::isActive()
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return false;
    return true;
}

void AdvancedTimer::stepBegin(IdStep id)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    Record r;
    r.time = CTime::getTime();
    r.type = Record::RSTEP_BEGIN;
    r.id = id;
    curRecords->push_back(r);
}

void AdvancedTimer::stepBegin(IdStep id, IdObj obj)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    Record r;
    r.time = CTime::getTime();
    r.type = Record::RSTEP_BEGIN;
    r.id = id;
    r.obj = obj;
    curRecords->push_back(r);
}

void AdvancedTimer::stepEnd  (IdStep id)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    if (syncCallBack) (*syncCallBack)(syncCallBackData);
    Record r;
    r.time = CTime::getTime();
    r.type = Record::RSTEP_END;
    r.id = id;
    curRecords->push_back(r);
}

void AdvancedTimer::stepEnd  (IdStep id, IdObj obj)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    Record r;
    r.time = CTime::getTime();
    r.type = Record::RSTEP_END;
    r.id = id;
    r.obj = obj;
    curRecords->push_back(r);
}

void AdvancedTimer::stepNext (IdStep prevId, IdStep nextId)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    Record r;
    if (syncCallBack) (*syncCallBack)(syncCallBackData);
    r.time = CTime::getTime();
    r.type = Record::RSTEP_END;
    r.id = prevId;
    curRecords->push_back(r);
    r.type = Record::RSTEP_BEGIN;
    r.id = nextId;
    curRecords->push_back(r);
}

void AdvancedTimer::step     (IdStep id)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    if (syncCallBack) (*syncCallBack)(syncCallBackData);
    Record r;
    r.time = CTime::getTime();
    r.type = Record::RSTEP;
    r.id = id;
    curRecords->push_back(r);
}

void AdvancedTimer::step     (IdStep id, IdObj obj)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    if (syncCallBack) (*syncCallBack)(syncCallBackData);
    Record r;
    r.time = CTime::getTime();
    r.type = Record::RSTEP;
    r.id = id;
    r.obj = obj;
    curRecords->push_back(r);
}

void AdvancedTimer::valSet(IdVal id, double val)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    Record r;
    r.time = CTime::getTime();
    r.type = Record::RVAL_SET;
    r.id = id;
    r.val = val;
    curRecords->push_back(r);
}

void AdvancedTimer::valAdd(IdVal id, double val)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    Record r;
    r.time = CTime::getTime();
    r.type = Record::RVAL_ADD;
    r.id = id;
    r.val = val;
    curRecords->push_back(r);
}

// API using strings instead of Id, to remove the need for Id creation when no timing is recorded

void AdvancedTimer::stepBegin(const char* idStr)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    stepBegin(IdStep(idStr));
}

void AdvancedTimer::stepBegin(const char* idStr, const char* objStr)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    stepBegin(IdStep(idStr), IdObj(objStr));
}

void AdvancedTimer::stepBegin(const char* idStr, const std::string& objStr)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    stepBegin(IdStep(idStr), IdObj(objStr));
}

void AdvancedTimer::stepEnd  (const char* idStr)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    stepEnd  (IdStep(idStr));
}

void AdvancedTimer::stepEnd  (const char* idStr, const char* objStr)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    stepEnd  (IdStep(idStr), IdObj(objStr));
}

void AdvancedTimer::stepEnd  (const char* idStr, const std::string& objStr)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    stepEnd  (IdStep(idStr), IdObj(objStr));
}

void AdvancedTimer::stepNext (const char* prevIdStr, const char* nextIdStr)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    stepNext (IdStep(prevIdStr), IdStep(nextIdStr));
}

void AdvancedTimer::step     (const char* idStr)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    step     (IdStep(idStr));
}

void AdvancedTimer::step     (const char* idStr, const char* objStr)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    step     (IdStep(idStr), IdObj(objStr));
}

void AdvancedTimer::step     (const char* idStr, const std::string& objStr)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    step     (IdStep(idStr), IdObj(objStr));
}

void AdvancedTimer::valSet(const char* idStr, double val)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    valSet(IdVal(idStr),val);
}

void AdvancedTimer::valAdd(const char* idStr, double val)
{
    helper::vector<Record>* curRecords = getCurRecords();
    if (!curRecords) return;
    valAdd(IdVal(idStr),val);
}

void TimerData::clear()
{
    nbIter = 0;
    steps.clear();
    stepData.clear();
    vals.clear();
    valData.clear();
}

void TimerData::process()
{
    if (records.empty()) return;
    ++nbIter;
    if (nbIter == 0) return; // do not keep stats on very first iteration

    ctime_t t0 = records[0].time;
    //ctime_t last_t = 0;
    int level = 0;
    for (unsigned int ri = 0; ri < records.size(); ++ri)
    {
        const Record& r = records[ri];
        ctime_t t = r.time - t0;
        //last_t = r.time;
        if (r.type == Record::REND || r.type == Record::RSTEP_END) --level;
        switch (r.type)
        {
        case Record::RNONE:
            break;
        case Record::RBEGIN:
        case Record::RSTEP_BEGIN:
        case Record::RSTEP:
        {
            AdvancedTimer::IdStep id;
            if (r.type != Record::RBEGIN) id = AdvancedTimer::IdStep(r.id);
            if (stepData.find(id) == stepData.end())
                steps.push_back(id);
            StepData& data = stepData[id];
            data.level = level;
            if (data.lastIt != nbIter)
            {
                data.lastIt = nbIter;
                data.tstart += t;
                ++data.numIt;
            }
            data.lastTime = t;
            ++data.num;
            break;
        }
        case Record::REND:
        case Record::RSTEP_END:
        {
            AdvancedTimer::IdStep id;
            if (r.type != Record::REND) id = AdvancedTimer::IdStep(r.id);
            StepData& data = stepData[id];
            if (data.lastIt == nbIter)
            {
                ctime_t dur = t - data.lastTime;
                data.ttotal += dur;
                data.ttotal2 += dur*dur;
                if (data.num == 1 || dur > data.tmax) data.tmax = dur;
                if (data.num == 1 || dur < data.tmin) data.tmin = dur;
            }
            break;
        }
        case Record::RVAL_SET:
        case Record::RVAL_ADD:
        {
            AdvancedTimer::IdVal id = AdvancedTimer::IdVal(r.id);
            if (valData.find(id) == valData.end())
                vals.push_back(id);
            ValData& data = valData[id];
            if (r.type == Record::RVAL_SET || (data.lastIt != nbIter))
            {
                // update vmin and vmax
                if (data.num == 1 || data.vtotalIt < data.vmin) data.vmin = data.vtotalIt;
                if (data.num == 1 || data.vtotalIt > data.vmax) data.vmax = data.vtotalIt;
            }
            if (data.lastIt != nbIter)
            {
                data.lastIt = nbIter;
                data.vtotalIt = r.val;
                data.vtotal += r.val;
                data.vtotal2 += r.val*r.val;
                ++data.numIt;
                ++data.num;
            }
            else if (r.type == Record::RVAL_SET)
            {
                data.vtotalIt = r.val;
                data.vtotal += r.val;
                data.vtotal2 += r.val*r.val;
                ++data.num;
            }
            else
            {
                data.vtotalIt += r.val;
                data.vtotal += r.val;
                data.vtotal2 += r.val*r.val;
            }
            break;
        }
        }

        if (r.type == Record::RBEGIN || r.type == Record::RSTEP_BEGIN) ++level;
    }

    for (unsigned int vi=0; vi < vals.size(); ++vi)
    {
        AdvancedTimer::IdVal id = vals[vi];
        ValData& data = valData[id];
        if (data.num > 0)
        {
            // update vmin and vmax
            if (data.num == 1 || data.vtotalIt < data.vmin) data.vmin = data.vtotalIt;
            if (data.num == 1 || data.vtotalIt > data.vmax) data.vmax = data.vtotalIt;
        }
    }
}

void printVal(std::ostream& out, double v)
{
    if (v < 0)
    {
        v = -v;
        v += 0.005;
        long long i = (long long)floor(v);
        if (i >= 10000)
        {
            v += 0.495;
            i = (long long)floor(v);
            out << "-" << i;
            if (i < 100000)
                out << ' ';
        }
        else if (i >= 1000)
        {
            v += 0.045;
            i = (long long)floor(v);
            int dec = (int)floor((v-i)*10);
            out << '-' << i;
            if (dec == 0)
                out << "  ";
            else
                out << '.' << dec;
        }
        else
        {
            int dec = (int)floor((v-i)*100);
            long long m = 100;
            while (i < m && m > 1)
            {
                out << ' ';
                m /= 10;
            }
            out << '-' << i;
            if (dec == 0)
                out << "   ";
            else if (dec < 10)
                out << ".0" << dec;
            else
                out << '.' << dec;
        }
    }
    else
    {
        v += 0.005;
        long long i = (long long)floor(v);
        if (i >= 100000)
        {
            v += 0.495;
            i = (long long)floor(v);
            out << i;
            if (i < 1000000)
                out << ' ';
        }
        else if (i >= 10000)
        {
            v += 0.045;
            i = (long long)floor(v);
            int dec = (int)floor((v-i)*10);
            out << i;
            if (dec == 0)
                out << "  ";
            else
                out << '.' << dec;
        }
        else
        {
            int dec = (int)floor((v-i)*100);
            long long m = 1000;
            while (i < m && m > 1)
            {
                out << ' ';
                m /= 10;
            }
            out << i;
            if (dec == 0)
                out << "   ";
            else if (dec < 10)
                out << ".0" << dec;
            else
                out << '.' << dec;
        }
    }
};

void printNoVal(std::ostream& out)
{
    out << "       ";
};

void printVal(std::ostream& out, double v, int niter)
{
    if (niter == 0)
        printNoVal(out);
    else
        printVal(out, v/niter);
}

void printTime(std::ostream& out, ctime_t t, int niter=1)
{
    static ctime_t timer_freq = CTime::getTicksPerSec();
    printVal(out, 1000.0 * (double)t / (double)(niter*timer_freq));
}

void TimerData::print()
{
    static ctime_t tmargin = CTime::getTicksPerSec() / 100000;
    std::ostream& out = std::cout;
    out << "==== " << id << " ====\n\n";
    if (!records.empty())
    {
        out << "Trace of last iteration :\n";
        ctime_t t0 = records[0].time;
        ctime_t last_t = 0;
        int level = 0;
        for (unsigned int ri = 1; ri < records.size(); ++ri)
        {
            const Record& r = records[ri];
            out << "  * ";
            if (ri > 0 && ri < records.size()-1 && r.time <= last_t + tmargin)
            {
                printNoVal(out);
                out << "   ";
            }
            else
            {
                printTime(out, r.time - t0);
                out << " ms";
                last_t = r.time;
            }
            out << " ";
            if (r.type == Record::REND || r.type == Record::RSTEP_END) --level;
            for (int l=0; l<level; ++l)
                out << "  ";
            switch(r.type)
            {
            case Record::RNONE:
                out << "NONE";
                break;
            case Record::RSTEP_BEGIN:
                out << "> begin " << AdvancedTimer::IdStep(r.id);
                if (r.obj)
                    out << " on " << AdvancedTimer::IdObj(r.obj);
                break;
            case Record::RSTEP_END:
                out << "< end   " << AdvancedTimer::IdStep(r.id);
                if (r.obj)
                    out << " on " << AdvancedTimer::IdObj(r.obj);
                break;
            case Record::RSTEP:
                out << "- step  " << AdvancedTimer::IdStep(r.id);
                if (r.obj)
                    out << " on " << AdvancedTimer::IdObj(r.obj);
                break;
            case Record::RVAL_SET:
                out << ": var   " << AdvancedTimer::IdVal(r.id);
                out << "  = " << r.val;
                break;
            case Record::RVAL_ADD:
                out << ": var   " << AdvancedTimer::IdVal(r.id);
                out << " += " << r.val;
                break;
            case Record::REND:
                out << "END";
                break;
            default:
                out << "UNKNOWN RECORD TYPE" << (int)r.type;
            }
            out << std::endl;
            if (r.type == Record::RBEGIN || r.type == Record::RSTEP_BEGIN) ++level;
        }
    }
    if (!steps.empty())
    {
        out << "\nSteps Duration Statistics (in ms) :\n";
        out << " LEVEL\t START\t  NUM\t   MIN\t   MAX\t MEAN\t  DEV\t TOTAL\tPERCENT\tID\n";
        ctime_t ttotal = stepData[AdvancedTimer::IdStep()].ttotal;
        for (unsigned int s=0; s<steps.size(); ++s)
        {
            StepData& data = stepData[steps[s]];
            printVal(out, data.level);
            out << '\t';
            printTime(out, data.tstart, data.numIt);
            out << '\t';
            printVal(out, data.num, (s == 0) ? 1 : nbIter);
            out << '\t';
            printTime(out, data.tmin);
            out << '\t';
            printTime(out, data.tmax);
            out << '\t';
            double mean = (double)data.ttotal / data.num;
            printTime(out, (ctime_t)mean);
            out << '\t';
            printTime(out, (ctime_t)(sqrt((double)data.ttotal2/data.num - mean*mean)));
            out << '\t';
            printTime(out, data.ttotal, (s == 0) ? 1 : nbIter);
            out << '\t';
            printVal(out, 100.0*data.ttotal / (double) ttotal);
            out << '\t';
            if (s == 0)
                out << "TOTAL";
            else
            {
                for(int ii=0; ii<data.level; ii++) out<<".";  // indentation to show the hierarchy level
                out << steps[s];
            }
            out << std::endl;
        }
    }
    if (!vals.empty())
    {
        out << "\nValues Statistics :\n";
        out << " NUM\t  MIN\t  MAX\t MEAN\t  DEV\t TOTAL\tID\n";
        for (unsigned int s=0; s<vals.size(); ++s)
        {
            ValData& data = valData[vals[s]];
            printVal(out, data.num, nbIter);
            out << '\t';
            printVal(out, data.vmin);
            out << '\t';
            printVal(out, data.vmax);
            out << '\t';
            double mean = data.vtotal / data.num;
            printVal(out, mean);
            out << '\t';
            printVal(out, sqrt(data.vtotal2/data.num - mean*mean) );
            out << '\t';
            printVal(out, data.vtotal, nbIter);
            out << '\t';
            out << vals[s];
            out << std::endl;
        }
    }

    out << "\n==== END ====\n";
    out << std::endl;
}

void TimerData::print(std::ostream& result)
{
    //static ctime_t tmargin = CTime::getTicksPerSec() / 100000;
    std::ostream& out = result;
    out << "Timer: " << id << "\n";
    if (!steps.empty())
    {
        //out << "\nSteps Duration Statistics (in ms) :\n";
        out << " LEVEL      START       NUM         MIN        MAX       MEAN       DEV        TOTAL     PERCENT     ID\n";
        ctime_t ttotal = stepData[AdvancedTimer::IdStep()].ttotal;
        for (unsigned int s=0; s<steps.size(); ++s)
        {
            StepData& data = stepData[steps[s]];
            printVal(out, data.level);
            out << "    ";
            printTime(out, data.tstart, data.numIt);
            out << "    ";
            printVal(out, data.num, (s == 0) ? 1 : nbIter);
            out << "    ";
            printTime(out, data.tmin);
            out << "    ";
            printTime(out, data.tmax);
            out << "    ";
            double mean = (double)data.ttotal / data.num;
            printTime(out, (ctime_t)mean);
            out << "    ";
            printTime(out, (ctime_t)(sqrt((double)data.ttotal2/data.num - mean*mean)));
            out << "    ";
            printTime(out, data.ttotal, (s == 0) ? 1 : nbIter);
            out << "    ";
            printVal(out, 100.0*data.ttotal / (double) ttotal);
            out << "    ";
            if (s == 0)
                out << "TOTAL";
            else
            {
                for(int ii=0; ii<data.level; ii++) out<<".";  // indentation to show the hierarchy level
                out << steps[s];
            }
            out << std::endl;
        }
    }
    if (!vals.empty())
    {
        out << "\nValues Statistics :\n";
        out << " NUM\t  MIN\t  MAX\t MEAN\t  DEV\t TOTAL\tID\n";
        for (unsigned int s=0; s<vals.size(); ++s)
        {
            ValData& data = valData[vals[s]];
            printVal(out, data.num, nbIter);
            out << '\t';
            printVal(out, data.vmin);
            out << '\t';
            printVal(out, data.vmax);
            out << '\t';
            double mean = data.vtotal / data.num;
            printVal(out, mean);
            out << '\t';
            printVal(out, sqrt(data.vtotal2/data.num - mean*mean) );
            out << '\t';
            printVal(out, data.vtotal, nbIter);
            out << '\t';
            out << vals[s];
            out << std::endl;
        }
    }

    //out << "\n==== END ====\n";
    out << std::endl;
}

}

}
