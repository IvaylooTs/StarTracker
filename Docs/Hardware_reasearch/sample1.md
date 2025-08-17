System boot sequence
![[BootCycle.drawio(2).png]]

After the system boots, linux service is started. The service is designed to reboot in case of crash of the program it represents - in this case a bash script that reloads the main program.
The bash script is used to ensure that the main program can reboot and call different exit conditions from the developers. The service is to allow overview of the process and to ensure initialization of the boot script and its recovery.

The main program kills every thread that uses port 5000 and 6789, to free slots for Flask server and Websocket. Then the main program runs. There are few cases to consider:
- Exit code 2 - reboots the whole system
- Exit code 3 - shutdowns the whole system
- Any other exit code - reboot program. We use 23 to represent reboot called from the developer interface.
- Crash of main program - reboot program
- Crash of bash boot script - Service reloads

#### Logger
Small script is used to record data with custom commands. We save everything in `logs/` folder, where each file is timestamped.  Custom commands are made for displaying information in the terminal and the file.
Format is "Time | Message type | Information"
Message types are:
- Info
- Success
- Warning
- Error
- Fatal error
