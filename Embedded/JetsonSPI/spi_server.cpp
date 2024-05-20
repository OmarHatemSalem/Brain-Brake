/* The port number is passed as an argument */
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include "jetsonGPIO.h"

using namespace std;

jetsonTX2GPIONumber CLK = gpio388;
jetsonTX2GPIONumber MOSI = gpio298;
jetsonTX2GPIONumber SS = gpio480;

void delay(int t)
{
    std::this_thread::sleep_for(std::chrono::microseconds(t));
}

void setUpSPI() 
{
    // export pins
    gpioExport(CLK);
    gpioExport(MOSI);
	gpioExport(SS);

    // set pins direction
    gpioSetDirection(CLK, outputPin);
	gpioSetDirection(MOSI, outputPin);
	gpioSetDirection(SS, outputPin);

    // set clk to high
	gpioSetValue(CLK, high);

	// set select to high
	gpioSetValue(SS, high);
}

void SPIWrite(u_int8_t c) 
{

    int8_t i;

	gpioSetValue(CLK, high);
	gpioSetValue(SS, low);

    for(i=0; i<8; ++i) 
	{
		gpioSetValue(MOSI, (c & (1 << i)? high : low));
		delay(1);
		gpioSetValue(CLK, low);
		delay(1);
		gpioSetValue(CLK, high);
    }
   
	gpioSetValue(CLK, high);
	gpioSetValue(SS, high);
}

void error(const char *msg)
{
    perror(msg);
    exit(1);
}

void get_values(char buffer[], float &speed, float &distance)
{
    // get values bytes from buffer
    char speed_bytes[4] = {buffer[0], buffer[1], buffer[2], buffer[3]};
    char distance_bytes[4] = {buffer[4], buffer[5], buffer[6], buffer[7]};

    // convert bytes to float (assumes representation is the same on both machines)
    memcpy(&speed, &speed_bytes, sizeof(speed));
    memcpy(&distance, &distance_bytes, sizeof(distance));
}

void spi_send_buffer(char buffer[], int n_bytes)
{
    for (int i = 0; i < n_bytes; i++)
    {
        SPIWrite(buffer[i]);
    }
}

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        fprintf(stderr, "ERROR, no port provided\n");
        exit(1);
    }
    int portno = atoi(argv[1]);

    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
        error("ERROR opening socket");

    struct sockaddr_in serv_addr;
    bzero((char *)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);

    if (bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
        error("ERROR on binding");

    listen(sockfd, 5);

    struct sockaddr_in cli_addr;
    socklen_t clilen = sizeof(cli_addr);

    int newsockfd = accept(sockfd, (struct sockaddr *)&cli_addr, &clilen);
    if (newsockfd < 0)
        error("ERROR on accept");

    printf("server: got connection from %s port %d\n", inet_ntoa(cli_addr.sin_addr), ntohs(cli_addr.sin_port));

    char buffer[256];
    int n_bytes;

    while (1)
    {
        bzero(buffer, 256);
        n_bytes = read(newsockfd, buffer, 255);

		clock_t begin = clock();
		{
		    if (n_bytes < 0)
		        error("ERROR reading from socket");
		    
		    if (n_bytes == 0)
		        break;

		    spi_send_buffer(buffer, n_bytes);
		}
		clock_t end = clock();
		double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

        float speed, distance;
        get_values(buffer, speed, distance);

        printf("Speed: %f\n", speed);
        printf("Distance: %f\n", distance);
		printf("Time take: %f\n", time_spent);
    }
    close(sockfd);
    return 0;
}
