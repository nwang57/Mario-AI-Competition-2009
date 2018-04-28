# Mario Environment

Mario enviroment is built by java but it supports a ServerAgent that can connect with any other agent written in different languages. It privides a Python client interface to extract the game information.

## Set up Server

A small bug in the original competition code [Mario Competition](https://github.com/rictic/Mario-AI-Competition-2009)

In `ToolsConfigurator.java`, at line 107 we need to change the `evaluationOptions.getAgent().getName()` to `evaluationOptions.getAgentName()` to avoid null pointer reference.

Then we use `ant` to build the project in the root directiory.

Run following commands to launch the server
```
cd classes
java -cp .:../lib/asm-all-3.3.jar ch.idsia.scenarios.Main -ag ServerAgent
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

Use the following command to lanuch human agent client

`python src/python/competition/ipymario.py`

Human can use wasd and j, k to control mario

## Q-learning agent

`python src/python/competition/ipymario.py --agent learning --n [num episodes] --model [dqn|ddqn]`

## A2C agent

`python src/python/competition/ipymario.py --agent pg --n [num episodes] --pg_model [model path]`

 ## Global options

 The default FPS is 24 so the delay is 41. We can change the delay in `MarioComponent.adjustFPS()`. 


## Game options

`-tl 100` set time limit to 100
`-vis off` turn off visualization
`-mm [0|1|2]` change mario mode
