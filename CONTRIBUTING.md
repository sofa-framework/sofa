# CONTRIBUTING GUIDELINES


## Pull-requests

- Pull-request must be properly submitted
    - Intelligible description of the PR: description of the **issue solved**, or the **feature added**
	- one PR = one logical modification (one fix or one new feature)
- Commit must build **successfully** on Jenkins for all platforms (compilation + tests + examples)
- **Examples** (at least one) must be provided showing the new feature
	- examples/Components
- **Test** required for each new component or if an issue is fixed
    - SofaTest
    - in myplugin/myplugin_test
    - possibly for each template e.g. Vec3d, Rigid3d ?
- Code must be **documented** in a Doxygen compliant way and in english
    - global presentation in header
    - description of each function and parameter/Data


## General rules

### Base rules
The base rules are Google C++ Style Guide: https://google.github.io/styleguide/cppguide.html   
All rules below **replace** the corresponding base rules.  
For any subject not mentionned below, please refer to the base.

### Naming
Identifiers must respect the following conventions

- Class names are in UpperCamelCase
- Function names are in lowerCamelCase()
- Namespaces names are in lowercase
- Variables names are in lowerCamelCase and must express their use more than their type.  
Exceptions:  mathematical objects like `Matrix M`, local iterators variables like `i`, `j`, `k`  and usual symbols like `x` for positions or `v` for velocities.
- Data member variables names must begin with `d_`
- Link member variables names (e.g. SingleLink) must begin with `l_`
- Other member variables names must begin with `m_ `(exception: this is not mandatory for PODs as well as public attributes)
- Names for booleans variables must answer a question: `m_isRed`, `m_hasName`
- C++ files must must have the extension .h, .cpp, or .inl
- Files that define a class should have the same name as the class and should contain only one class.
- Each library name should be prefixed with `Sofa`.

### Formatting
- Special characters like TAB and page break must be avoided.
- Indentation must use **4 spaces** everywhere (C++ and Python), but there must be no indentation for namespaces
- Braces use the **Allman style**: the opening brace associated with a control statement is on the next line, indented to the same level as the control statement, and statements within the braces are indented to the next level.

    ```cpp
    while (x == y)
    {
        something();
        somethingElse();
    }
    finalThing();
    ```
- A space character should be used in the following situations:
    - After C++ reserved words: `if (true)`
    - Around binary operators: `a + b`
    - After commas: `doSomething(a, b, c)`
    - After semicolons in for statements: `for (unsigned int i = 0; i < container.size(); i++)`
    ```cpp
    for (unsigned int i = 0; i < container.size(); ++i )
    {
        if ((a + b) > c)
        {
            doSomething(a, b, c);
        }
    }
    ```

- Template declarations are split on two lines
    ```cpp
    template<class T>
    static void dynamicCast(T*& ptr, Base* b);
    ```

### Coding
- You should try to use as few `#include` directive as possible.
- You should limit as much as possible the amount of code in included files (*.h, *.inl).
- All definitions should reside in source files. The header files should declare an interface only.
- Variables should be initialized when they are declared.
- You should use `const` profusely.
- You should use `assert` profusely.
- You must avoid the `using` directive in header files (.h and .inl): ~~`using namespace foo;`~~
- You should declare automatic variables only when you need them (not before).
- You must always initialize pointers, either to the address of something, or to `NULL`


## SOFA specific rules
- Tricky code should not be commented but rewritten! In general, the use of comments should be minimized by making the code self-documenting by appropriate name choices and an explicit logical structure.
- All the code under development must be tagged `SOFA_DEV`
- The use of magic numbers in the code should be avoided. Numbers other than 0 and 1 should be declared as named constants instead.
- All internal data, needed by your component, and that can’t be recomputed must be put inside a `Data` or a `DataPtr`. This way, your component can be saved. Also, this `Data` will be automatically displayed inside the GUI.
- Use `sout`, `serr`, `sendl` instead of `cout`, `cerr`, `endl` in SOFA Components.
- `serr` will automatically display inside the console a message with a warning, the name of the component, and its class.  
If you modify the component in the graph, you will see a tabulation named `Warnings` with the log of all the `serr` done by the component.
- `sout` will display inside the console a message ONLY if the Data f_printLog is set to true.  
If you modify the component in the graph, you will see a tabulation named `Outputs` with the log of all the `sout` done by the component
- Use `sofa::helper::vector` and `sofa::helper::set` instead of `std::vector` and `std::set`
- Only use `sofa::simulation::tree::GNode` when you need to directly use access to the children or the parent of the node. If not, use the more generic `sofa::simulation::Node`

