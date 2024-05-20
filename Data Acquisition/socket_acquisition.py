import UnicornPy
import numpy as np
import socket
import pickle
import struct
import time

def main():
    # Specifications for the data acquisition.
    #-------------------------------------------------------------------------------------
    TestsignaleEnabled = False;
    # Try out different framelengths
    FrameLength = 1;
    # FrameLength = 10;
    AcquisitionDurationInSeconds = 3000;
    
    print("Unicorn Acquisition Example")
    print("---------------------------")
    print()

    try:
        # Get available devices.
        #-------------------------------------------------------------------------------------

        # Get available device serials.
        deviceList = UnicornPy.GetAvailableDevices(True)

        if len(deviceList) <= 0 or deviceList is None:
            raise Exception("No device available.Please pair with a Unicorn first.")

        # Print available device serials.
        print("Available devices:")
        i = 0
        for device in deviceList:
            print("#%i %s" % (i,device))
            i+=1

        # Request device selection.
        print()
        # deviceID = int(input("Select device by ID #"))
        deviceID = 0
       
        
        if deviceID < 0 or deviceID > len(deviceList):
            raise IndexError('The selected device ID is not valid.')

        # Open selected device.
        #-------------------------------------------------------------------------------------
        print()
        print("Trying to connect to '%s'." %deviceList[deviceID])
        device = UnicornPy.Unicorn(deviceList[deviceID])
        print("Connected to '%s'." %deviceList[deviceID])
        print()



        # Initialize acquisition members.
        #-------------------------------------------------------------------------------------
        numberOfAcquiredChannels= device.GetNumberOfAcquiredChannels()
        configuration = device.GetConfiguration()
        print(configuration)


        # Print acquisition configuration
        print("Acquisition Configuration:");
        print("Sampling Rate: %i Hz" %UnicornPy.SamplingRate);
        print("Frame Length: %i" %FrameLength);
        print("Number Of Acquired Channels: %i" %numberOfAcquiredChannels);
        print("Data Acquisition Length: %i s" %AcquisitionDurationInSeconds);
        print();

        # Allocate memory for the acquisition buffer.
        receiveBufferBufferLength = FrameLength * numberOfAcquiredChannels * 4
        receiveBuffer = bytearray(receiveBufferBufferLength)

        try:
            # Start data acquisition.
            #-------------------------------------------------------------------------------------
            device.StartAcquisition(TestsignaleEnabled)
            print("Data acquisition started.")

            # Calculate number of get data calls.
            
            # We dont need this bec. we want to have continuous input 
            numberOfGetDataCalls = int(AcquisitionDurationInSeconds * UnicornPy.SamplingRate / FrameLength);
        
            # Limit console update rate to max. 25Hz or slower to prevent acquisition timing issues.                   
            consoleUpdateRate = int((UnicornPy.SamplingRate / FrameLength) / 25.0);
            if consoleUpdateRate == 0:
                consoleUpdateRate = 1

            # Send data on socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                HOST = "192.168.1.2"  # Standard loopback interface address (localhost)
                PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
                s.bind((HOST, PORT))
                s.listen()
                print("Waiting for a connection")
                conn, addr = s.accept()
                print(f"Connection from {addr}")
                # Acquisition loop.
                #-------------------------------------------------------------------------------------
                
                # change this to while (true)

                for i in range (0,numberOfGetDataCalls):
                    try:
                        # Receives the configured number of samples from the Unicorn device and writes it to the acquisition buffer.
                        device.GetData(FrameLength,receiveBuffer,receiveBufferBufferLength)

                        # Convert receive buffer to numpy float array 
                        data = np.frombuffer(receiveBuffer, dtype=np.float32, count=numberOfAcquiredChannels * FrameLength)
                        data = np.reshape(data, (FrameLength, numberOfAcquiredChannels))
                        
                        # Create a dedicated thread for sending the data: try pyqt threading 
                        # chunk the data
                        if data is not None:
                            print(data.shape)

                            # maybe pickling takes time
                            data1 = pickle.dumps(data)
                            conn.sendall(struct.pack(">L", len(data1)) + data1)
                            # Similarly for latest_image2 if needed
                            time.sleep(0.01)
                    except Exception as e:
                        print(f"An error occurred: {e}")
                    # finally:
                    #     print("Closing connection")
                    #     conn.close()                
                    # Update console to indicate that the data acquisition is running.
                    if i % consoleUpdateRate == 0:
                        print('.',end='',flush=True)

            # Stop data acquisition.
            #-------------------------------------------------------------------------------------
            device.StopAcquisition();
            print()
            print("Data acquisition stopped.");

        except UnicornPy.DeviceException as e:
            print(e)
        except Exception as e:
            print("An unknown error occured. %s" %e)
        finally:
            # release receive allocated memory of receive buffer
            del receiveBuffer

            # Close device.
            #-------------------------------------------------------------------------------------
            del device
            print("Disconnected from Unicorn")

    except Unicorn.DeviceException as e:
        print(e)
    except Exception as e:
        print("An unknown error occured. %s" %e)

    input("\n\nPress ENTER key to exit")

#execute main
main()
