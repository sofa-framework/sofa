#ifndef SOFA_HELPER_ADVANCEDTIMER_H
#define SOFA_HELPER_ADVANCEDTIMER_H

#include <sofa/helper/helper.h>
#include <string>
#include <iostream>

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
        Id() : id(0) {}

        /// An Id is constructed from a string and appears like one after, without actually storing a string
        Id(const std::string& s);

        /// An Id is constructed from a string and appears like one after, without actually storing a string
        Id(const char* s);

        /// This constructor should be used only if really necessary
        Id(unsigned int id) : id(id) {}

        /// Any operation requiring an int can be used on an id using this conversion
        operator unsigned int() const { return id; }

        /// Any operation requiring a string can be used on an id using this conversion
        operator std::string() const;

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

    static void clear();
    static void begin(IdTimer id);
    static void end  (IdTimer id);

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

    class StepVar
    {
    public:
        IdStep id;
        IdObj obj;
        StepVar(IdStep id) : id(id)
        {
            stepBegin(id);
        }
        StepVar(IdStep id, IdObj obj) : id(id), obj(obj)
        {
            stepBegin(id, obj);
        }
        template<class T>
        StepVar(IdStep id, T* obj) : id(id), obj(IdObj(obj->getName()))
        {
            stepBegin(id, obj);
        }
        ~StepVar()
        {
            if (!obj)
                stepEnd(id);
            else
                stepEnd(id, obj);
        }
    };

    static void valSet(IdVal id, double val);
    static void valAdd(IdVal id, double val);
};

}

}

#endif
