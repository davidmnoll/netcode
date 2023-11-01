import subprocess
import os

def start_service(command, path=None):
    cwd = os.getcwd()
    if path:
        os.chdir(path)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.chdir(cwd)
    return process

try:
    # Start the frontend
    frontend_command = 'make run-fe'
    frontend_process = start_service(frontend_command, './')
    
    # Start the backend
    # backend_process = start_service('make run-be', './')
    
    frontend_process.wait()
    # backend_process.wait()

except KeyboardInterrupt:
    # Handle Ctrl+C gracefully
    print("\nTerminating processes...")
    frontend_process.terminate()
    # backend_process.terminate()

except Exception as e:
    print(f"Error: {e}")
    frontend_process.terminate()
    # backend_process.terminate()
