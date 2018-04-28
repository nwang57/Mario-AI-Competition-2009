package ch.idsia.tools.tcp;

import ch.idsia.agents.Agent;
import ch.idsia.agents.controllers.BasicMarioAIAgent;
import ch.idsia.benchmark.mario.environments.Environment;
import ch.idsia.tools.EvaluationInfo;

import edu.stanford.cs229.agents.MarioState;

import java.io.IOException;

/**
 * Created by IntelliJ IDEA.
 * User: Sergey Karakovskiy
 * Date: Apr 30, 2009
 * Time: 9:43:27 PM
 * Package: ch.idsia.tools.Network
 */

public class ServerAgent extends BasicMarioAIAgent implements Agent
{
    Server server = null;
    private int port;
    private TCP_MODE tcpMode = TCP_MODE.SIMPLE_TCP;
    private MarioState currState = new MarioState();

    public ServerAgent(int port, boolean enable)
    {
        super("ServerAgent");
        this.port = port;
        if (enable)
        {
            createServer(port);
        }
    }

    public ServerAgent(Server server, boolean isFastTCP)
    {
        super("ServerAgent");
        this.server = server;
        this.tcpMode = (isFastTCP) ? TCP_MODE.FAST_TCP : TCP_MODE.SIMPLE_TCP;
    }

    public String getName()
    {
        return this.name + ((server == null) ? "" : server.getClientName());
    }

    public void setFastTCP(boolean isFastTCP)
    {
        this.tcpMode = (isFastTCP) ? TCP_MODE.FAST_TCP : TCP_MODE.SIMPLE_TCP;
    }

    // A tiny bit of singletone-like concept. Server is created ones for each egent. Basically we are not going
    // To create more than one ServerAgent at a run, but this flexibility allows to add this feature with certain ease.
    private void createServer(int port) {
        this.server = new Server(port, Environment.numberOfKeys);
//        this.name += server.getClientName();
    }

    public boolean isAvailable()
    {
        return (server != null) && server.isClientConnected();
    }

    public void reset()
    {
        isFinished = false;
        action = new boolean[Environment.numberOfKeys];
        if (server == null)
            this.createServer(port);
        currState = new MarioState();
    }

    private void sendRawObservation(Environment observation)
    {
//        byte[][] levelScene = observation.getLevelSceneObservation();
        // MERGED
        byte[][] mergedObs = observation.getMergedObservationZZ(1, 1);
        System.out.println("length " + mergedObs.length);
        String tmpData = "O " +
                observation.isMarioAbleToJump() + " " + observation.isMarioOnGround();
        for (int x = 0; x < mergedObs.length; ++x)
        {
            for (int y = 0; y < mergedObs.length; ++y)
            {
                tmpData += " " + (mergedObs[x][y]);
            }
        }
        tmpData += " " + observation.getMarioFloatPos()[0]
                 + " " + observation.getMarioFloatPos()[1];
        
        float[] enemiesFloatPoses = observation.getEnemiesFloatPos();
        for (int i = 0; i < enemiesFloatPoses.length; ++i)
            tmpData += " " + enemiesFloatPoses[i];

        server.sendSafe(tmpData);
        // TODO: StateEncoderDecoder.Encode.Decode.  zip, gzip do not send mario position. zero instead for better compression.
    }

    public void integrateObservation(Environment environment)
    {
        //    sendRawObservation(environment);

            sendObservation(environment);
        
    }


    private void sendObservation(Environment observation)
    {
        // if (this.tcpMode == TCP_MODE.SIMPLE_TCP)
        // {
            //this.sendRawObservation(observation);
            this.currState.update(observation);

            // long bitData = this.currState.getStateNumber();
            String obs = this.currState.get_obs();
            obs = "X " + obs;
            float reward = this.currState.calculateReward();
            obs += " " + Math.round(reward);
            // System.out.println("reward is " + reward);
            // System.out.println(obs);
            server.sendSafe(obs);
        // }
        // else if (this.tcpMode == TCP_MODE.FAST_TCP)
        // {
        //     this.sendBitmapObservation(observation);
        // }
    }

    public void setFinished() {
        server.sendSafe("FIN");
    }

//     private void sendBitmapObservation(Environment observation)
//     {
        
//         String tmpData =  "E" +
//                           (observation.isMarioAbleToJump() ? "1" : "0")  +
//                           (observation.isMarioOnGround() ? "1" : "0") +
//                           observation.getBitmapLevelObservation();
// //                          observation.getBitmapEnemiesObservation();
//         int check_sum = 0;
//         for (int i = 3; i < tmpData.length(); ++i)
//         {
//             char cur_char = tmpData.charAt(i);
//             if (cur_char != 0)
//             {
// //                System.out.print(i + " ");
// //                MathX.show(cur_char);
//                 check_sum += Integer.valueOf(cur_char);
//             }
//         }
//         if (tmpData.length() != /*125 - 61*/34)
//             try {
//                 throw new Exception("Pipetz!");
//             } catch (Exception e) {
//                 e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
//                 System.err.println(e.getMessage());
//             }
//         tmpData += " " + check_sum;
// //        System.out.println("tmpData size = " + tmpData.length());
//         server.sendSafe(tmpData);
//     }

    public void integrateEvaluationInfo(EvaluationInfo evaluationInfo)
    {
        String fitnessStr = "FIT " +
                evaluationInfo.marioStatus + " " +
                evaluationInfo.computeDistancePassed() + " " +
                evaluationInfo.timeLeft + " " +
                evaluationInfo.marioMode + " " +
                evaluationInfo.coinsGained + " ";
        server.sendSafe(fitnessStr);
    }

    private boolean[] receiveAction() throws IOException, NullPointerException
    {
        String data = server.recvSafe();
        boolean[] ret = new boolean[Environment.numberOfKeys];
        // while (data == null) {
        //     data = server.recvSafe();
        // }
        if (data == null || data.startsWith("reset"))
            return ret;
//        String s = "[";
        for (int i = 0; i < Environment.numberOfKeys; ++i)
        {
            ret[i] = (data.charAt(i) == '1');
//            s += data.charAt(i);
        }
//        s += "]";

//        System.out.println("ServerAgent: action received :" + s);
        return ret;
    }

    public boolean[] getAction()
    {
        try
        {
        //    sendRawObservation(observation);
            // sendObservation(observation);
            action = receiveAction();
        }
        catch (IOException e)
        {
            e.printStackTrace();
            System.out.println("I/O Communication Error");
            reset();
        }
        return action;
    }

    public void printAction(boolean[] action) {
        System.out.print("Server: ");
        for( boolean v : action) {
            System.out.print(v);
        }
        System.out.println("");
    }

}
