-- PurePod Worker Controller
-- Save as Application in Script Editor (File > Export > File Format: Application)

use AppleScript version "2.4"
use scripting additions

property workerPath : "/Users/kait/Dev/PurePod/worker"
property terminalWindowID : missing value

on run
	displayMainWindow()
end run

on displayMainWindow()
	set buttonPressed to button returned of (display dialog "PurePod Worker Controller

Click Start to run in background.
Click Debug to run with Terminal output." buttons {"Start", "Stop", "Quit"} default button "Start" with icon note)

	if buttonPressed is "Start" then
		showStartOptions()
		displayMainWindow()
	else if buttonPressed is "Stop" then
		stopWorker()
		displayMainWindow()
	else if buttonPressed is "Quit" then
		return
	end if
end displayMainWindow

on showStartOptions()
	set startOption to button returned of (display dialog "Start Mode" buttons {"Background", "Debug", "Cancel"} default button "Background" with icon note)

	if startOption is "Background" then
		startWorkerDetached()
	else if startOption is "Debug" then
		startWorkerDebug()
	end if
end showStartOptions

on startWorkerDetached()
	-- Check if already running
	set isRunning to isWorkerRunning()

	if isRunning then
		display dialog "Worker is already running!" buttons {"OK"} default button "OK" with icon caution
		return
	end if

	-- Start the worker detached in background (logs go to file)
	do shell script "cd " & quoted form of workerPath & " && source .venv/bin/activate && nohup celery -A worker.main worker --loglevel=info --pool=solo > worker.log 2>&1 &"

	display notification "Worker started (detached)" with title "PurePod"
end startWorkerDetached

on startWorkerDebug()
	-- Check if already running
	set isRunning to isWorkerRunning()

	if isRunning then
		display dialog "Worker is already running!" buttons {"OK"} default button "OK" with icon caution
		return
	end if

	-- Start the worker in a new Terminal window for debugging
	tell application "Terminal"
		activate
		set newWindow to do script "cd " & quoted form of workerPath & " && source .venv/bin/activate && celery -A worker.main worker --loglevel=info --pool=solo"
	end tell

	display notification "Worker started (debug mode)" with title "PurePod"
end startWorkerDebug

on stopWorker()
	set isRunning to isWorkerRunning()
	
	if not isRunning then
		display dialog "Worker is not running." buttons {"OK"} default button "OK" with icon note
		return
	end if
	
	-- Kill celery worker processes
	try
		do shell script "pkill -9 -f 'worker.main'"
		delay 1
		display notification "Worker stopped" with title "PurePod"
	on error
		display dialog "Failed to stop worker. Try manually closing the Terminal window." buttons {"OK"} default button "OK" with icon stop
	end try
end stopWorker

on viewLogs()
	set isRunning to isWorkerRunning()
	
	if not isRunning then
		display dialog "Worker is not running. Start it first to see logs." buttons {"OK"} default button "OK" with icon note
		return
	end if
	
	-- Just bring Terminal to front - logs are already visible there
	tell application "Terminal"
		activate
	end tell
end viewLogs

on isWorkerRunning()
	try
		do shell script "pgrep -f 'worker.main'"
		return true
	on error
		return false
	end try
end isWorkerRunning
