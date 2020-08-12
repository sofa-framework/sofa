SOFA launcher
=============

The tool *sofa-launcher* ease the scripting of **numerous SOFA simulations**. This can be done from XML or python scripts. To accelerate the processing of the simulations the script has the ability to run the simulation either: sequentially, in parallel or on a cluster.

There is two options to use it depending on your needs:

- You want to run a lot of simulation from you own python script. In this case you should look at the file "integration_example.py". This example shows how to start simulation in sequence, in parallel or using a cluster. The example also shows how all the simulation results are returned back to your script so you can implement your own number-crunching processing on them, e.g.: plotting graphs with *matplotlib*.

- You don't want to write your own python script but still want to start a lot a lot of simulations you should have a look at the sofa-launcher.py application.
Example:
```batch
./sofa-launcher example.json
```


This application is controlled via a json configuration file
```json
{
  "files": [
    "example.scn",
    "example.py"
  ],
  "variables": [
    {
      "GRAVITYXML": "1 2 3",
      "nbIterations": 1000
    },
    {
      "GRAVITYXML": "2 3 4",
      "nbIterations": 1000
    }
  ],
  "launcher": "parallel",
  "numjobs": 5,
  "resultsstyle": "results.thtml",
  "resultsfile": "results.html"
}
```
with:

- files: The sofa files with your scene
- variables: Some values in your scene can be changed automatically in each run of your scene  
             You can use any name as $MYVALUE...it depend on what you put in your scene.  
             in my example scene I replace the $GRAVITYXML with a different value at each run
- launcher: Indicate to launch the simulations in parallel (other choice are sequential or ssh)
- numjobs: with a maximum of 5 simulation in parallel
- resultsstyle: Name of the report file template, here html but it could be anything else (eg latex array)
- resultsfile: The file generated from the template and the results

NB:

- To run the distributed version of the launcher you need all hosts to share directories as well as being able to login using ssh+key (no password login).
- Depending on the files you are using in your scene you may need to protect certain symbols: see [https://pythonhosted.org/Cheetah/users_guide/](https://pythonhosted.org/cheetah/users_guide/parserinstructions.html)
