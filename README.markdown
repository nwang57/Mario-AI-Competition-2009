# Mario Environment

Mario enviroment is built by java but it supports a ServerAgent that can connect with any other agent written in different languages. It privides a Python client interface to extract the game information.

## Set up Server

A small bug in the original competition code [Mario Competition](https://github.com/rictic/Mario-AI-Competition-2009)

In `ToolsConfigurator.java`, at line 107 we need to change the `evaluationOptions.getAgent().getName()` to `evaluationOptions.getAgentName()` to avoid null pointer reference.

Then we use `ant` to build the project in the root directiory.

Run following commands to launch the server
```
cd classes
java ch.idsia.scenarios.MainRun -ag ServerAgent -server on
```

## Set up Client

Example:

`python src/python/competition/ipymario.py` to launch the client and run the agent.

`ServerAgent.java` has the serialization protocol and can inspect the environment data.

## Observation

`dataadaptor.py` implements the deserialization protocol and extract the raw info of the environment.

## Action

The key map is as follows:
 - Left: 0
 - Right: 1
 - Down: 2
 - Jump: 3
 - Speed: 4

## HumanAgent

Human can use wasd and j, k to control mario

 ## Global options

 The default FPS is 24 so the delay is 41. We can change the delay in `MarioComponent.adjustFPS()`. 