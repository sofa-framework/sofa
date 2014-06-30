#include <iostream>
#include <vector>
#include "test1.h"


int MyClass::functionVALID(){
	// This one is not defined in a header..
}

int main()
{ 
    // Invalid declaration, we should always initialize
    MyClass* cl2INVALID ;
    cl2INVALID=new MyClass() ;

    // Invalid pod declaration, we should always initialize
    int aINVALID;
    // This one is ok.
    int bVALID=0, cINVALID, dVALID=2;

    for(int iINVALID; iINVALID < 10; iINVALID++)
    {

    }

    for(int iVALID=10; iVALID< 10; iVALID++)
    {

    }


    //
    MyClass* clVALID=new MyClass() ;
    clVALID->functionInvalid() ;
}
