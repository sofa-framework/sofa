# Communication Plugin

## Installation

### QSerialPort Installation

Refer to - https://wiki.qt.io/Qt_Serial_Port for all the instruction regarding installation of QSerialPort installation.

It supports all major operating systems.

### VRPN Installation

Linux :

```
sudo apt-get install vrpn
```

Windows : 

Compile the library by yourself using this zip file : https://github.com/vrpn/vrpn/releases/download/v07.33/vrpn_07_33.zip

MacOS :

```
brew install vrpn
```

> Although, the vrpn installed through brew should work for most cases but if it does not, it can always be build manually.

Compile the library by yourself using this zip file : https://github.com/vrpn/vrpn/releases/download/v07.33/vrpn_07_33.zip

### LibOscpack installation
Please ensure oscpack version is >= 1.1.X. Do not use the default packages provided by ubuntu repository (1.0.X version).
You can find a fully working version here : http://ftp.debian.org/debian/pool/main/o/oscpack/?C=M;O=D

#### Linux :
```
wget http://ftp.debian.org/debian/pool/main/o/oscpack/liboscpack1_1.1.0-2_amd64.deb
sudo dpkg -i liboscpack1_1.1.0-2_amd64.deb
wget http://ftp.debian.org/debian/pool/main/o/oscpack/liboscpack-dev_1.1.0-2_amd64.deb
sudo dpkg -i liboscpack-dev_1.1.0-2_amd64.deb
```
#### Windows :
Compile the library by yourself using this zip file https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/oscpack/oscpack_1_1_0_RC2.zip

#### MacOS :
Compile the library by yourself using this zip file https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/oscpack/oscpack_1_1_0_RC2.zip

### ZMQ installation
Depending of your distribution, the package name can be different.

#### Linux : 

Ubuntu :
```
sudo apt-get install libzmq3-dev
```
Fedora : 
```
sudo dnf install libzmq-devel
```
#### Windows :
Compile the library by yourself using this zip file : https://github.com/zeromq/libzmq/releases/download/v4.2.3/zeromq-4.2.3.zip

#### MacOS :
Compile the library by yourself using this zip file : https://github.com/zeromq/libzmq/releases/download/v4.2.3/zeromq-4.2.3.zip

> Note : `zmq.hpp` is not created by default while compiling the library. While, building sofa if error occurs like `zmq.hpp not found`, manually paste the file in this location `/usr/local/include`. 

## How to use the components
To learn how to create a SOFA scene, please refer to the tutorial provided by the SOFA Modeler or this documentation.

Here is an example of how you can use the Communication component. In this example we want to send or receive sofa's data 

### ServerCommunication

ServerCommunication is an abstract class allowing users to create asynchronous communication class . Actually, there is three implementations of it. These three implementations use - 
* ZMQ
* OSC
* VRPN
* QSerial

ServerCommunication provides default DataFields :
* job -> "receiver" or "sender". Depends if you want to receive or send datas. Default value is "receiver"
* refreshRate -> an int. This is only related to the send part. For exemple if you set 2 a message will be sent every 2hz (aka 500ms). Default value is 30
* address -> an int. Define on which network address you want to send data. Default value is "127.0.0.1"
* port -> an int. Define on which network port you want to send data. Default value is 6000

OSC and ZMQ implementations have specifics DataFields :
* OSC
  * packetSize -> an int. Define size of OSC packets. Default value is 1024
* ZMQ
  * pattern -> "publish/subscribe" or "request/reply". It describe how zmq will works. Default value "publish/subscribe".

### Subscriber

Subscriber is a class needed to ServerCommunication. This will allow user to define which kind of message he want to send/receive and which sofa´s data he want to bind to.

Subscriber DataFields explanation : 
* subject -> a string. ServerCommunication will receive and send only subscribed subjects. Default value is ""
* communication -> a ServerCommunication link. A subscriber is attached to a serverCommunication
* target -> a BaseObject link. This object will be use to read/write data in it
* datas -> a string. A list of variables name. Existing or not inside the target

A serverCommunication should contains at least one subscriber.

### How to use ServerCommunication QSerialPort

#### Receive

```
<ServerCommunicationQSerial name="qserial" job="receiver" port="6000"/>
<CommunicationSubscriber name="sub1" communication="@qserial" subject="/colorLight" target="@light1" datas="color"/>
```

#### Send

```
<ServerCommunicationQSerial name="qserial" job="sender" port="6000"/>
<CommunicationSubscriber name="sub1" communication="@qserial" subject="/colorLight" target="@light1" datas="color"/>
```

### How to use ServerCommunication VRPN

#### Receive

```
<ServerCommunicationVRPN name="vrpn1" job="receiver" address="localhost"/>
<CommunicationSubscriber name="sub1" communication="@vrpn1" subject="testing" target="@light1" datas="aNewStringValue"/>
```

#### Send

```
<ServerCommunicationVRPN name="vrpn1" job="sender" address="localhost"/>
<CommunicationSubscriber name="sub1" communication="@vrpn1" subject="testing" target="@light1" datas="position"/>
```

### How to use ServerCommunication OSC

#### Receive
```
<ServerCommunicationOSC name="aServerCommunicationOSC" job="receiver" port="6000"/>
<CommunicationSubscriber name="subscriberOSC" communication="@aServerCommunicationOSC" subject="/test" target="@aServerCommunicationOSC" datas="x y"/>
```
#### Send
```
<ServerCommunicationOSC name="aServerCommunicationOSC" job="receiver" port="6000" refreshRate="2"/>
<CommunicationSubscriber name="subscriberOSC" communication="@aServerCommunicationOSC" subject="/test" target="@aServerCommunicationOSC" datas="x y"/>
```

### How to use ServerCommunication ZMQ

#### Receive
```
<ServerCommunicationZMQ name="aServerCommunicationZMQ" job="receiver" pattern="request/reply" port="6000"/>
<CommunicationSubscriber name="subscriberZMQ" communication="@aServerCommunicationZMQ" subject="/test" target="@aServerCommunicationZMQ" datas="x y"/>
```
#### Send
```
<ServerCommunicationZMQ name="aServerCommunicationZMQ" job="sender" pattern="request/reply" port="6000" refreshRate="2"/>
<CommunicationSubscriber name="subscriberZMQ" communication="@aServerCommunicationZMQ" subject="/test" target="@aServerCommunicationZMQ" datas="x y"/>
```
### Examples

A set of examples are availables in example plugin directory : [Examples](examples)

## How to implement a new network protocol

Implementing a new protocol is quite easy. Simply extend from ServerCommunication. Then you will have to implement some virtual methods such as :
* getFactoryInstance
* initTypeFactory
* sendData
* receiveData
* getArgumentType
* getArgumentValue
* defaultDataType

### Factory in a nutshell

If you try to fetch a non existing data the ServerCommunication will create it by asking a factory. 
Allowing the user to create sofa data from received data using factory is an elegant way to add flexibility and reusability. 
Two virtual function are related to the factory let's see how it has been implemented for OSC.

Header file : 
```
    //////////////////////////////// Factory OSC type /////////////////////////////////
    typedef CommunicationDataFactory OSCDataFactory;
    OSCDataFactory* getFactoryInstance() override;
    virtual void initTypeFactory() override;
    /////////////////////////////////////////////////////////////////////////////////
```

Cpp file : 
```
ServerCommunicationOSC::OSCDataFactory* ServerCommunicationOSC::getFactoryInstance(){
    static OSCDataFactory* s_localfactory = nullptr ;
    if(s_localfactory==nullptr)
        s_localfactory = new ServerCommunicationOSC::OSCDataFactory() ;
    return s_localfactory ;
}

void ServerCommunicationOSC::initTypeFactory()
{
    getFactoryInstance()->registerCreator("f", new DataCreator<float>());
    getFactoryInstance()->registerCreator("d", new DataCreator<double>());
    getFactoryInstance()->registerCreator("i", new DataCreator<int>());
    getFactoryInstance()->registerCreator("s", new DataCreator<std::string>());

    getFactoryInstance()->registerCreator("matrixf", new DataCreator<vector<float>>());
    getFactoryInstance()->registerCreator("matrixi", new DataCreator<vector<int>>());
    getFactoryInstance()->registerCreator("matrixd", new DataCreator<vector<double>>());
}
```

The getFactoryInstance function si responsible to return a DataFactory instance. In this case, a singleton. The initTypeFactory is the place where we will do the binding between receveived data type and sofa's type.
For example, using OSC, if we received a float his tag type will be "f". The equivalent in sofa is the primitive type float.
Then we bind "f" to float. In case of non existing data with type "f" the serverCommunication will create a sofa float data.

### Receive and send datas

```
    virtual void sendData() override;
    virtual void receiveData() override;
```

Those functions have to be implemented by the new protocols. Both are runs by the mother class inside a thread. This should be the place where you loop for sending or receiving datas. You can find examples in ZMQ, OSC and VRPN implementations.

For receiving and sending datas you will need to fetch them using the mother class function named fetchDataFromSenderBuffer and saveDatasToReceivedBuffer.

Example for sending datas : 

```
std::string ServerCommunicationZMQ::createZMQMessage(CommunicationSubscriber* subscriber, std::string argument)
{
    std::stringstream messageStr;
    BaseData* data = fetchDataFromSenderBuffer(subscriber, argument);
    if (!data)
        throw std::invalid_argument("data is null");
    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
const void* valueVoidPtr = data->getValueVoidPtr();
```
In this example we retrieve the data from a buffer using fetchDataFromSenderBuffer. The argument named argument is the argument name we want to fetch. You can find more details in ZMQ, OSC, VRPN and QSerial inl files.

Example for receiving datas :

```
```