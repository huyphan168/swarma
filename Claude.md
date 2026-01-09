Activation:
  conda activate swarm

  Dummy Solution Files (already exist)

  Located in swarm/submission_template/:
  - main.py - Entry point for the agent
  - drone_agent.py - DroneFlightController class with random action policy
  - agent_server.py - Cap'n Proto RPC server
  - agent.capnp - RPC schema

  Test File

  Located at tests/test_rpc.py - Tests your agent exactly like the validator does.

  Run tests with:
  conda activate swarm
  python tests/test_rpc.py swarm/submission_template/ --seed 42

  Add --gui to visualize the drone simulation.

  Key Constants

  - SIM_DT: 0.02s (50 Hz simulation)
  - HORIZON_SEC: 60s (time limit)
  - SPEED_LIMIT: 3.0 m/s
