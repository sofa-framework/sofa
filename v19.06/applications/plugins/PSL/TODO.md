PSL is in heavy development phase. It is the good moment to have an impact on the language design
by contributing.

#### Task to do
- unify PSL::templates and imported templates. (in pslengine.py)
- add saving
- add reactive programming
- support "bizarre" feature of XML
    - _ or NO to ignore a parameter
    - include (superseeded by Import)
    - objects (superseeded by Import)
- ADD TESTS !!! to validate both the execution engine & the loaders.
- when saving large chunk of data...use a dedicated opcode like 'extern('datafile')' to indicate we
  save them in a separated file (possible format are: json, xml, sofa string, python picked)
- it should be possible to save a file in a python pickled version so that it can be loaded faster.

#### DONE
- restore the sofa implicit links & template deduction mechanisme.
- add PSL/AliasComponent, PSL/ImportComponent
- move all the code into the PSL plugin (this means making python modules in pluging)
   - Python/Template to PSL/Template
   - Python/PythonComponent to PSL/PythonComponent
- add a test framework.
- add XML syntax
- add the pslloader & pslengine in the default python path.



#### Crazy **possibly unreleated things**
- add a component cache to reload the scene more quickly when live coding (ie mesh, texture, object)
- add a kind of previewer & editor(text field) to visualize template instance (with a template instance example)
