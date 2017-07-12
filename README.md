# Autonomous-Nav-for-Flying-Robots
This is the solution to the final homework of the edx course: Autonomous Navigation for Flying Robots

The solution contains python code used to navigate a quadrotor simulation to hit a series beacons autonomously using world markers and odometry measurements as a guide. In real life, these will come from various sensors on the quadrotor.

Key concepts involved include Kalman filters, visual localization, coordinate transformation and path planning

## Problem statement:
In this final exercise you have to fly again through a series of beacons. In contrast to previous exercises, we only provide you with the raw odometry measurements and marker observations from the quadrotor. You will have to implement:
 - marker placement and path planning (exercise week 1),
 - position controller (exercise week 4), and
 - state estimation (exercise week 6 and 7).

The goal of this exercise is to bring all together, so please make use of the solutions from these previous weeks. To support localization of the quadrotor you can place up to 30 markers in the world. Once you activated all beacons the simulation stops and you can submit your time. It has to be below 300s of simulation time  to get credits. We encourage you to post screenshots of your best time (and possibly your trajectory and map) in the discussion forum below.
