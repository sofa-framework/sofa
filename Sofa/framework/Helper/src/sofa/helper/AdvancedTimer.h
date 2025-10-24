/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/helper/config.h>
#include <sofa/helper/system/thread/thread_specific_ptr.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/type/vector.h>

#include <ostream>
#include <string>
#include <map>
#include <unordered_map>
#include <sofa/helper/narrow_cast.h>

namespace sofa::simulation
{
    class Node;
}

namespace sofa::helper
{

/**
  Advanced timer, meant to gather precise statistics for results in published papers.
  Not so advanced for now, but it will be...

  Usage examples :

  * When all computations start (i.e., Simulation::step):
    AdvancedTimer::begin("Animate");

  * When all computations stop (i.e., Simulation::step):
    AdvancedTimer::end("Animate");

  * Using a local variable to automatically call end when current instruction block (i.e. method) ends :
    AdvancedTimer::TimerVar("Animate");

  * When a part of the computation starts:
    AdvancedTimer::stepBegin("Collision");

  * When a part of the computation stops:
    AdvancedTimer::stepEnd("Collision");

  * Both operations combined:
    AdvancedTimer::stepNext("Collision", "Mechanical");

  * Using a local variable to automatically call stepEnd when current instruction block (i.e. method) ends :
    AdvancedTimer::StepVar("UpdateMapping");

  * Specifying the object being processed:
    AdvancedTimer::StepVar("Collision", objPtr);

  * When a noteworthy milestone happens:
    AdvancedTimer::step("Event1");

  * When a noteworthy value must be stored:
    AdvancedTimer::valSet("contacts",nbContacts);

  * When a noteworthy value must be accumulated:
    AdvancedTimer::valAdd("dofs",mstate->getSize());

  * When reloading/resetting the simulation:
    AdvancedTimer::clear();


  The produced stats will looks like:

  ==== Animate ====

  Trace of last iteration :
    *     0 ms > begin Collision
    *            : var   nbCM = 10
    *    10 ms   > begin BP
    *    20 ms   < end   NP
    *            > begin NP
    *   120 ms   < end   NP
    *            : var   nbContacts = 100
    *            > begin Response
    *   150 ms   < end   Response
    *          < end   Collision
    *          > begin Mechanical
    *          > begin CGSolve on Torus1
    *            : var   dofs += 300
    ...
    *   434 ms END

  Steps Duration Statistics (in ms) :
    LEVEL START   NUM     MEAN   MAX     TOTAL   ID
    0     0       100     222.2  546.3   22220   TOTAL
    1     0       1       80.5   120.7   80.5    Collision
    2     0       1       7.2    8.4     7.2     BP
    2     7.2     0.95    65.4   104.8   62.3    NP
    2     69.5    1       11.0   13.7    11.0    Response
    1     80.5    1       131.1  308.9   131.1   Mechanical
    2     80.5    10      13.1   45.7    131.0   CGSolve
    ...

  Values Statistics :
    MIN     MAX     MEAN    ID
    10      10      10      nbCM
    0       1230    420.3   nbContacts
    5000    5000    5000    dofs

  ==== END ====

 */

class Record
{
public:
    sofa::helper::system::thread::ctime_t time;
    enum Type { RNONE, RBEGIN, REND, RSTEP_BEGIN, RSTEP_END, RSTEP, RVAL_SET, RVAL_ADD } type;
    std::string label;
    unsigned int id;
    unsigned int obj;
    double val;
    Record() : type(RNONE), id(0), obj(0), val(0) {}
};

class StepData
{
public:
    int level;
    int num, numIt;
    sofa::helper::system::thread::ctime_t tstart;
    sofa::helper::system::thread::ctime_t tmin;
    sofa::helper::system::thread::ctime_t tmax;
    sofa::helper::system::thread::ctime_t ttotal;
    sofa::helper::system::thread::ctime_t ttotal2;
    int lastIt;
    sofa::helper::system::thread::ctime_t lastTime;
    std::string label;
    StepData() : level(0), num(0), numIt(0), tstart(0), tmin(0), tmax(0), ttotal(0), ttotal2(0), lastIt(-1), lastTime(0) {}
};

class SOFA_HELPER_API AdvancedTimer
{
public:

    template<class Base>
    class Id : public Base
    {
    public:
        /** Internal class used to generate IDs. */
        class IdFactory : public Base
        {
        protected:

            /// the list of the id names. the Ids are the indices in the vector
            std::vector<std::string> idsList;
            std::unordered_map<std::string, unsigned int> idsMap;

            IdFactory()
            {
                idsMap.insert({"0", 0});
                idsList.push_back("0");
            }
            
        public:
            
            /**
               @return the Id corresponding to the name of the id given in parameter
               If the name isn't found in the list, it is added to it and return the new id.
            */
            static unsigned int getID(const std::string& name)
            {
                if (name.empty())
                    return 0;
                IdFactory& idfac = getInstance();

                const auto it = idfac.idsMap.find(name);
                if (it == idfac.idsMap.end())
                {
                    const auto idsMapSize = sofa::helper::narrow_cast<unsigned int>(idfac.idsMap.size());
                    idfac.idsMap.insert(it, {name, idsMapSize});
                    idfac.idsList.push_back(name);
                    return idsMapSize;
                }

                return it->second;
            }

            static std::size_t getLastID()
            {
                return getInstance().idsList.size()-1;
            }

            /// return the name corresponding to the id in parameter
            static std::string getName(unsigned int id)
            {
                if (id < getInstance().idsList.size())
                    return getInstance().idsList[id];
                else
                    return "";
            }

            /// return the instance of the factory. Creates it if doesn't exist yet.
            static IdFactory& getInstance()
            {
                static thread_local IdFactory factory;
                return factory;
            }
        };

        Id() : id(0) {}

        /// An Id is constructed from a string and appears like one after, without actually storing a string
		Id(const std::string& s): id(0)
        {
            if (!s.empty())
                id = IdFactory::getID(s);
        }

        /// An Id is constructed from a string and appears like one after, without actually storing a string
		Id(const char* s): id(0)
        {
            if (s && *s)
                id = IdFactory::getID(std::string(s));
        }

        /// This constructor should be used only if really necessary
        Id(unsigned int idTimer) : id(idTimer) {}

        /// Any operation requiring an int can be used on an id using this conversion
        operator unsigned int() const { return id; }

        /// Any operation requiring a string can be used on an id using this conversion
        operator std::string() const
        {
            if (id == 0) return std::string("0");
            else return IdFactory::getName(id);
        }

        bool operator==(const Id<Base>& t) const { return id == t.id; }
        bool operator!=(const Id<Base>& t) const { return id != t.id; }
        bool operator<(const Id<Base>& t) const { return id < t.id; }
        bool operator>(const Id<Base>& t) const { return id > t.id; }
        bool operator<=(const Id<Base>& t) const { return id <= t.id; }
        bool operator>=(const Id<Base>& t) const { return id >= t.id; }
        bool operator!() const { return !id; }

        friend std::ostream& operator<<(std::ostream& o, const Id<Base>& t)
        {
            return o << (std::string)t;
        }

        friend std::istream& operator>>(std::istream& i, Id<Base>& t)
        {
            std::string s;
            i >> s;
            t = Id<Base>(s);
            return i;
        }

    protected:
        unsigned int id;
    };

    class Timer {};
    class Step {};
    class Val {};
    class Obj {};
    typedef Id<Timer> IdTimer;
    typedef Id<Step> IdStep;
    typedef Id<Val> IdVal;
    typedef Id<Obj> IdObj;

    enum outputType
    {
        STDOUT,
        LJSON,
        JSON,
        GUI
    };


    static bool isEnabled(IdTimer id);
    static void setEnabled(IdTimer id, bool val);
    static int  getInterval(IdTimer id);
    static void setInterval(IdTimer id, int val);

	/**
	 * @brief convertOutputType convert a string to the output type
	 * @param type std::string, output type name (example : "json")
	 * @return outputType, output type enum
	 */
	static AdvancedTimer::outputType convertOutputType(std::string type);

    /**
     * @brief setOutputType Set the outputType for the given AdvancedTimer.
     * @param id IdTimer, id of the timer
	 * @param type std::string, output type name (example : "json")
     **/
	static void setOutputType(IdTimer id, const std::string& type);

	/**
	 * @brief getOutputType Get the outputType for the given AdvancedTimer.
	 * @param id idTimer, id of the timer
	 * @return the output type
	 */
	static AdvancedTimer::outputType getOutputType(IdTimer id);


    /**
     * @brief getTimeAnalysis Return the result of the AdvancedTimer
     * @param id IdTimer, id of the timer
     * @param time double, current time (from the context)
     * @param time dt, current delta time or dt (from the context)
     * @return The timer value in JSON
     */
    static std::string getTimeAnalysis(IdTimer id, double time, double dt);

    /**
     * @brief getSteps Return the vector of IDStep of the AdvancedTimer given execution
     * @param id IdTimer, id of the timer
     * @param processData bool, if true, will force timer data to be processed
     * @return The timer steps iterator inside a vector
     */
    static type::vector<AdvancedTimer::IdStep> getSteps(IdTimer id, bool processData = false);

    /**
     * @brief getStepData Return the map of StepData of the AdvancedTimer given execution
     * @param id IdTimer, id of the timer
     * @param processData bool, if true, will force timer data to be processed
     * @return The timer StepData of each timer step inside a map
     */
    static std::map<AdvancedTimer::IdStep, StepData> getStepData(IdTimer id, bool processData = false);

    /**
     * @brief getRecords the vector of Record of the AdvancedTimer given execution id.
     * @param id IdTimer, id of the timer
     * @return The timer full records inside a vector of Record
     */
    static type::vector<Record> getRecords(IdTimer id);

    /**
     * @brief clearDatato clear a specific Timer Data
     * @param id IdTimer, id of the timer
     */
    static void clearData(IdTimer id);

    static void clear();
    static void begin(IdTimer id);
    static void end  (IdTimer id);
    static void end  (IdTimer id, std::ostream& result);


    /**
     * @brief end Override fo the end method in which you can use JSON or old format
     * @param id IdTimer, the id of the used timer
     * @param time double, current time (from the context)
     * @param time dt, current delta time or dt (from the context)
     * @return std::string, the output if JSON format is set
     */
    static std::string end(IdTimer id, double time, double dt);


    static bool isActive();

    class TimerVar
    {
    public:
        IdTimer id;
        TimerVar(IdTimer idtimer) : id(idtimer)
        {
            begin(id);
        }
        ~TimerVar()
        {
            end(id);
        }
    };

    static void stepBegin(IdStep id);
    static void stepBegin(IdStep id, IdObj obj);
    template<class T>
    static void stepBegin(IdStep id, T* obj)
    {
        stepBegin(id, IdObj(obj->getName()));
    }
    static void stepEnd  (IdStep id);
    static void stepEnd  (IdStep id, IdObj obj);
    template<class T>
    static void stepEnd  (IdStep id, T* obj)
    {
        stepEnd(id, IdObj(obj->getName()));
    }
    static void stepNext (IdStep prevId, IdStep nextId);
    static void step     (IdStep id);
    static void step     (IdStep id, IdObj obj);
    template<class T>
    static void step     (IdStep id, T* obj)
    {
        step(id, IdObj(obj->getName()));
    }

    // API using strings instead of Id, to remove the need for Id creation when no timing is recorded

    static void begin(const char* idStr);
    static void end(const char* idStr);

    static void stepBegin(const char* idStr);
    static void stepBegin(const char* idStr, const char* objStr);
    static void stepBegin(const char* idStr, const std::string& objStr);
    template<class T>
    static void stepBegin(const char* idStr, T* obj)
    {
        stepBegin(idStr, obj->getName());
    }
    static void stepEnd  (const char* idStr);
    static void stepEnd  (const char* idStr, const char* objStr);
    static void stepEnd  (const char* idStr, const std::string& objStr);
    template<class T>
    static void stepEnd  (const char* idStr, T* obj)
    {
        stepEnd(idStr, obj->getName());
    }
    static void stepNext (const char* prevIdStr, const char* nextIdStr);
    static void step     (const char* idStr);
    static void step     (const char* idStr, const char* objStr);
    static void step     (const char* idStr, const std::string& objStr);
    template<class T>
    static void step     (const char* idStr, T* obj)
    {
        step(idStr, obj->getName());
    }

    class StepVar
    {
    public:
        const IdStep id;
        const char* idStr;
        const IdObj obj;
        const char* objStr;
        StepVar(IdStep idTimer) : id(idTimer), idStr(nullptr), objStr(nullptr)
        {
            stepBegin(id);
        }
        StepVar(const char* idTimerStr) : idStr(idTimerStr), objStr(nullptr)
        {
            stepBegin(idStr);
        }
        StepVar(IdStep idTimer, IdObj objTimer) : id(idTimer), idStr(nullptr), obj(objTimer), objStr(nullptr)
        {
            stepBegin(id, obj);
        }
        StepVar(const char* idTimerStr, const char* objTimerStr) : idStr(idTimerStr), objStr(objTimerStr)
        {
            stepBegin(idStr, objStr);
        }
        template<class T>
        StepVar(IdStep idTimer, T* objTimer) : id(idTimer), idStr(nullptr), obj(IdObj(objTimer->getName())), objStr(nullptr)
        {
            stepBegin(id, obj);
        }
        template<class T>
        StepVar(const char* idTimerStr, T* objTimer) : idStr(idTimerStr), objStr(objTimer->getName().c_str())
        {
            stepBegin(idStr, objStr);
        }
        ~StepVar()
        {
            if (obj)
                stepEnd(id, obj);
            else if (id)
                stepEnd(id);
            else if (objStr)
                stepEnd(idStr, objStr);
            else if (idStr)
                stepEnd(idStr);
        }
    };

    static void valSet(IdVal id, double val);
    static void valAdd(IdVal id, double val);

    static void valSet(const char* idStr, double val);
    static void valAdd(const char* idStr, double val);


    typedef void (*SyncCallBack)(void* userData);
    static std::pair<SyncCallBack,void*> setSyncCallBack(SyncCallBack cb, void* userData = nullptr);

};

#if !defined(SOFA_HELPER_ADVANCEDTIMER_CPP)
extern template class SOFA_HELPER_API AdvancedTimer::Id<AdvancedTimer::Timer>;
extern template class SOFA_HELPER_API AdvancedTimer::Id<AdvancedTimer::Step>;
extern template class SOFA_HELPER_API AdvancedTimer::Id<AdvancedTimer::Obj>;
extern template class SOFA_HELPER_API AdvancedTimer::Id<AdvancedTimer::Val>;
#endif

}

