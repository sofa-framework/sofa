Sofa-launcher.

This tool ease the starting of large number of sofa simulations 
from python script (but not only). To accelerate the 
processing of the simulations the script has the ability to run the 
simulation in sequence, in parallel as well as on
a cluster of machines.

There is two options to use it depending on your needs.

You want to run a lot of simulation from you own python script. In this 
case you should look at the file "integration_example.py". This example 
shows how to start simulation in sequence, in parallel or using a cluster. 
The example also show how all the simulations results are returned back to 
your script so you can implement your own number-crunching processing on 
them eg: plotting graphs with matplotlib.

In case you don't want to write your own python script but still want to 
start a lot a lot of simulations you should have a look at the
sofa-launcher.py application.
eg: ./sofa-launcher example.json

This application is controlled via a json configuration file
{
         "files"       : ["example.scn", "example.py"],    /// The sofa files with your scene
         "variables"   : [{                                /// Some values in your scene can be changed automatically in each run of your scene
         "GRAVITYXML"      : "1 2 3",                      /// You can use any name as $MYVALUE...it depend on what you put in your scene.
                                 "nbIterations" : 1000     /// in my example scene I replace the $GRAVITYXML with a different value at each run
                         },
                         {
                                 "GRAVITYXML"      : "2 3 4",
                                 "nbIterations" : 1000
                         }
                         ],
         "launcher"    : "parallel",            /// Indicate to launch the simulations in parallel (other choice are sequential or ssh)
         "numjobs"     : 5,                     /// with a maximum of 5 simulation in parallel
         "resultsstyle" : "results.thtml",      /// Name of the report file template (here html but it could be anything else (eg latex array:)))
         "resultsfile"  : "results.html",       /// The file generated from the template and the results
}

To run the distributed version of the launcher you need all hosts to share
directories as well as being able to login using ssh+key (no password login).

...............

Depending on the files you are using in your scene you may need to protect certain symbols:
https://pythonhosted.org/cheetah/users_guide/parserinstructions.html




