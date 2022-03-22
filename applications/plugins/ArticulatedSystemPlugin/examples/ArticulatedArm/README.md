# Articulated Arm 

![](images/robot.png)

## Requirements 

- [SofaPython3](https://github.com/sofa-framework/SofaPython3) plugin for SOFA

### Optional

To use a GUI to control the robot you need to install [tkinter](https://docs.python.org/3/library/tkinter.html) for python. For instance on Ubuntu:

`sudo apt-install python-tk`

If you don't want to install tkinter just comment the following line in `robot.py`:

```python
    # Comment this if you don't want to use the GUI
    robot.addObject(RobotGUI(robot=robot))
```

## How to run the simulation

```bash
runSofa robot.py
```

