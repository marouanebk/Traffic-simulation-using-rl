import os
import sys
import traci

# Import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

def run():

    sumo_binary = "sumo-gui"  
    sumo_cmd = [sumo_binary, 
                "-c", "project.sumocfg",
                "--start","--delay", "50"]  

    traci.start(sumo_cmd)

    step = 0
    while step < 3600:  
        traci.simulationStep()

        
        step += 1

    # Close TraCI
    traci.close()

if __name__ == "__main__":
    run()