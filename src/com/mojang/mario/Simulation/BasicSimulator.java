package com.mojang.mario.Simulation;

import com.mojang.mario.MarioComponent;
import com.mojang.mario.GlobalOptions;
import com.mojang.mario.Tools.EvaluationInfo;

/**
 * Created by IntelliJ IDEA.
 * User: Sergey Karakovskiy
 * Date: Apr 7, 2009
 * Time: 2:27:48 PM
 * Package: com.mojang.mario.Simulation
 */

public class BasicSimulator implements ISimulation
{
    SimulationOptions simulationOptions = null;
    private MarioComponent marioComponent;

    public BasicSimulator(SimulationOptions simulationOptions)
    {
        GlobalOptions.VizualizationOn = simulationOptions.isVisualization();
        this.marioComponent = simulationOptions.getMarioComponent();
        this.setSimulationOptions(simulationOptions);
    }

    private MarioComponent prepareMarioComponent()
    {
        marioComponent.setAgent(simulationOptions.getAgent().reset());
        return marioComponent;
    }

    public void setSimulationOptions(SimulationOptions simulationOptions)
    {
        this.simulationOptions = simulationOptions;
    }

    public EvaluationInfo simulateOneLevel()
    {
        prepareMarioComponent();
        marioComponent.startLevel(simulationOptions.getLevelRandSeed(), simulationOptions.getLevelDifficulty()
                                 , simulationOptions.getLevelType(), simulationOptions.getLevelLength());
        marioComponent.setPaused(simulationOptions.isPauseWorld());
        return marioComponent.run1();
    }
}