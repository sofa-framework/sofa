/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_HELPER_ADVANCEDTIMER_H
#define SOFA_HELPER_ADVANCEDTIMER_H
#include <sofa/helper/helper.h>
#include <sofa/simulation/Simulation.h>


#include <iostream>
#include <string>
#include <vector>


namespace sofa
{

namespace helper
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

  * When reloading/reseting the simulation:
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

class SOFA_HELPER_API AdvancedTimer
{
public:

    template<class Base>
    class Id : public Base
    {
    public:
        /** Internal class used to generate IDs. */
        class SOFA_HELPER_API IdFactory : public Base
        {
        protected:

            /// the list of the id names. the Ids are the indices in the vector
            std::vector<std::string> idsList;

            IdFactory()
            {
                idsList.push_back(std::string("0")); // ID 0 == "0" or empty string
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
                std::vector<std::string>::iterator it = idfac.idsList.begin();
                unsigned int i = 0;

                while (it != idfac.idsList.end() && (*it) != name)
                {
                    ++it;
                    i++;
                }

                if (it!=idfac.idsList.end())
                    return i;
                else
                {
                    idfac.idsList.push_back(name);
                    return i;
                }
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
                static IdFactory instance;
                return instance;
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
        Id(unsigned int id) : id(id) {}

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
        JSON
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
     * @param node Node*, pointeur on a node to get the scene simulation context
     * @return The timer value in JSON
     */
    static std::string getTimeAnalysis(IdTimer id, simulation::Node* node);


    static void clear();
    static void begin(IdTimer id);
    static void end  (IdTimer id);
    static void end  (IdTimer id, std::ostream& result);


    /**
     * @brief end Ovveride fo the end method in which you can use JSON or old format
     * @param id IdTimer, the id of the used timer
     * @param node Node*, node used to get the scene cotext
     * @return std::string, the output if JSON format is set
     */
    static std::string end(IdTimer id, simulation::Node* node);

    static bool isActive();

    class TimerVar
    {
    public:
        IdTimer id;
        TimerVar(IdTimer id) : id(id)
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
        StepVar(IdStep id) : id(id), idStr(NULL), objStr(NULL)
        {
            stepBegin(id);
        }
        StepVar(const char* idStr) : idStr(idStr), objStr(NULL)
        {
            stepBegin(idStr);
        }
        StepVar(IdStep id, IdObj obj) : id(id), idStr(NULL), obj(obj), objStr(NULL)
        {
            stepBegin(id, obj);
        }
        StepVar(const char* idStr, const char* objStr) : idStr(idStr), objStr(objStr)
        {
            stepBegin(idStr, objStr);
        }
        template<class T>
        StepVar(IdStep id, T* obj) : id(id), idStr(NULL), obj(IdObj(obj->getName())), objStr(NULL)
        {
            stepBegin(id, obj);
        }
        template<class T>
        StepVar(const char* idStr, T* obj) : idStr(idStr), objStr(obj->getName().c_str())
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
    static std::pair<SyncCallBack,void*> setSyncCallBack(SyncCallBack cb, void* userData = NULL);

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_HELPER_ADVANCEDTIMER_CPP)
extern template class SOFA_HELPER_API AdvancedTimer::Id<AdvancedTimer::Timer>;
extern template class SOFA_HELPER_API AdvancedTimer::Id<AdvancedTimer::Step>;
extern template class SOFA_HELPER_API AdvancedTimer::Id<AdvancedTimer::Obj>;
extern template class SOFA_HELPER_API AdvancedTimer::Id<AdvancedTimer::Val>;
#endif


/// Scoped (RAII) AdvancedTimer to simplify a basic usage
struct ScopedAdvancedTimer {

    const char* message;

    ScopedAdvancedTimer( const char* message )
    : message( message )
    {
        AdvancedTimer::stepBegin( message );
    }

    ~ScopedAdvancedTimer()
    {
        AdvancedTimer::stepEnd( message );
    }
};


}

}

#endif
