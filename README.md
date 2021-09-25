# Energy-Saver-(Street-Light-Controller)-Power-Usage-Prediction
Description


A company has implemented more than 100 energy efficiency projects and is now remotely managing more than 100MW load including lakhs of street lamps and tens of thousands of pumps.


CCMS or the Standalone Street Light Controller is a control panel with comprehensive protection, control and monitoring station for a group of street lights. It includes a Class1.0 metering unit and communicates to the SMART web server with GSM/GPRS connection.


**
Key Highlights of Street Light controller:

Over / under voltage protection

Overload protection

Short circuit protection

Auto rectification for nuisance MCB Trips

Tolerant to input voltage fluctuations

Surge protection up to 40 KA

Astronomical / Photocell / Configurable ON/OFF timings

Event notification for faults

Problem Statement:
We are tasked with predicting the number of units consumed by each street light controller. The data is received from the IoT device which is deployed in the various states in India.

Mapping the real-world problem to an ML problem:

Type of Machine Learning Problem

Supervised Learning:

It is a regression problem, for a given data we need to predict the energy consumption of the street light controller

Train and Test Construction

We build train and test by randomly splitting in the ratio of 70:30 or 80:20 whatever we choose as we have sufficient points to work with.
Importing the Necessary Libraries
