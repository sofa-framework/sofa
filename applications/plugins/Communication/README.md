#Communication Plugin

## Installation
### LibOscpack installation
Please ensure oscpack version is >= 1.1.X. Do not use the default packages provided by ubuntu repository (1.0.X version).
You can find a fully working version here : http://ftp.debian.org/debian/pool/main/o/oscpack/?C=M;O=D

Ubuntu :
```
wget http://ftp.debian.org/debian/pool/main/o/oscpack/liboscpack1_1.1.0-2_amd64.deb
sudo dpkg -i liboscpack1_1.1.0-2_amd64.deb
wget http://ftp.debian.org/debian/pool/main/o/oscpack/liboscpack-dev_1.1.0-2_amd64.deb
sudo dpkg -i liboscpack-dev_1.1.0-2_amd64.deb
```

### ZMQ installation
Depending of your distribution, the package name can be different.

Ubuntu :
```
sudo apt-get install libzmq3-dev
```
Fedora : 
```
sudo dnf install libzmq-devel
```
Windows :
Compile the lib by yourself using this zip file : https://github.com/zeromq/libzmq/releases/download/v4.2.3/zeromq-4.2.3.zip

## How to use the components
To learn how to create a SOFA scene, please refer to the tutorial provided by the SOFA Modeler or this documentation.

Here is an example of how you can use the Communication component. In this example we want to send or receive sofa's data 

### ServerCommunication

ServerCommunication is an abstract class allowing users to create asynchronous communication class . Actually, there is two implementations of it. One using the OSC protocol and the other one using ZMQ protocol.

ServerCommunication provides default DataFields :
* job -> "receiver" or "sender". Depends if you want to receive or send datas. Default value is "receiver"
* refreshRate -> an int. This is only related to the send part. For exemple if you set 2 a message will be sent every 2hz (aka 500ms). Default value is 30
* address -> an int. Define on which network address you want to send data. Default value is "127.0.0.1"
* port -> an int. Define on which network port you want to send data. Default value is 6000

OSC and ZMQ implementations have specifics DataFields :
* OSC
  * packetSize -> an int. Define size of OSC packets. Default value is 1024
* ZMQ
  * pattern -> "publish/subscribe" or "request/reply". It describe how zmq will works. Default value "publish/subscribe"

### Subscriber

Subscriber is a class needed to ServerCommunication. This will allow user to define which kind of message he want to send/receive and which sofa´s data he want to bind to.

Subscriber DataFields explanation : 
* subject -> a string. ServerCommunication will receive and send only subscribed subjects. Default value is ""
* communication -> a ServerCommunication link. A subscriber is attached to a serverCommunication
* target -> a BaseObject link. This object will be use to read/write data in it
* datas -> a string. A list of variables name. Existing or not inside the target

A serverCommunication should contains at least one subscriber.

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

TODO send/receive explanations
