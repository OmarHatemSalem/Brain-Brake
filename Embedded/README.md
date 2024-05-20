# Embedded Hardware
This page documents the embedded part of the project.

## Components
- Orbitty Carrier for NVIDIA® Jetson™ TX2
- 2 Nucleo STM32F303RE Boards
- 2 MCP2551 CAN-BUS transceivers
- 2 120 Ohm resistors.

## Connectivity
Each MCP2551 transceiver is connected to an STM32 board. The CAN High and CAN Low wires of the bus are terminated with 120 Ohm resistors. One SPI interface of one of the STMs is connected to the four GPIO pins of the Jetson board.

## Communication
The jetson and STM communicate through SPI, the two STMs communicate through the CAN bus.

## Code Description
The folders BrakeingECU and CommunicationController contain two projects developed using STM32CubeIDE 1.14.1. The BrakingECU project contains the functionality of receiving a message of 8 bytes from SPI. It is then using COM and CAN communicates the message to the other end of the CAN bus.

The CommunicationController contains the project that receives the CAN frame from the CAN bus and then applies brakes based on the signals.

The Jetson doesn't have SPI natively, so we simulate an SPI master sender using GPIO pins. We do so in C++ to ensure minimum latency. However, most of the models run in python. That is why we provide a python client that connects to the spi server through providing the same port with running the server and the client. The code runs in linux which is the OS on the jetson.
