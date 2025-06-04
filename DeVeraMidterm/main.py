import os
import sys
import time
from getpass import getpass

# Dictionary to simulate user data for authentication
user_db = {"admin": "password123"}

# Mock file system storage
file_system = {}

# Mock process management storage
processes = []

# Notification log
notifications = []


# Function to display notifications
def show_notifications():
    if notifications:
        print("\n[Notifications]")
        for n in notifications:
            print(n)
        notifications.clear()
    else:
        print("\nNo new notifications.")


# Basic Security Functionality
def authenticate_user():
    username = input("Username: ")
    print(f"Entered username: {username}")  # Debugging print statement
    password = input("Password: ")  # Changed to input() for testing purposes
    print(f"Entered password: {password}")  # Debugging print statement

    if username in user_db and user_db[username] == password:
        print("Login successful!")
        notifications.append(f"User '{username}' logged in at {time.ctime()}")
        return True
    else:
        print("Invalid credentials!")
        notifications.append(f"Failed login attempt for '{username}' at {time.ctime()}")
        return False


# File System Functions
def create_file(filename, content):
    file_system[filename] = content
    notifications.append(f"File '{filename}' created at {time.ctime()}")


def read_file(filename):
    if filename in file_system:
        return file_system[filename]
    else:
        notifications.append(f"Failed file access '{filename}' at {time.ctime()}")
        return "File not found."


def delete_file(filename):
    if filename in file_system:
        del file_system[filename]
        notifications.append(f"File '{filename}' deleted at {time.ctime()}")
        print(f"File '{filename}' deleted.")
    else:
        print(f"File '{filename}' does not exist.")
        notifications.append(f"Failed deletion attempt for '{filename}' at {time.ctime()}")

# Process Management Functions
def start_process(name):
    process_id = len(processes) + 1
    process_info = {"id": process_id, "name": name, "start_time": time.ctime()}
    processes.append(process_info)
    notifications.append(f"Process '{name}' started with PID {process_id} at {time.ctime()}")


def list_processes():
    print("\n[Running Processes]")
    for process in processes:
        print(f"PID: {process['id']}, Name: {process['name']}, Started: {process['start_time']}")


# Main Program
def main():
    print("Welcome to the Python OS Simulation!")

    # Authenticate user
    if not authenticate_user():
        print("Exiting system due to failed authentication.")
        return

    # Main loop
    while True:
        input("\nPress enter to proceed...")  # Prompt to press any key

        print("\nChoose an option:")
        print("1. Create a file")
        print("2. Read a file")
        print("3. Delete a file")
        print("4. Start a process")
        print("5. List running processes")
        print("6. Show notifications")
        print("7. Logout")

        choice = input("Enter choice: ")

        if choice == "1":
            filename = input("Enter file name: ")
            content = input("Enter file content: ")
            create_file(filename, content)
            print(f"File '{filename}' created.")

        elif choice == "2":
            filename = input("Enter file name to read: ")
            print(read_file(filename))

        elif choice == "3":
            filename = input("Enter file name to delete: ")
            delete_file(filename)

        elif choice == "4":
            process_name = input("Enter process name: ")
            start_process(process_name)
            print(f"Process '{process_name}' started.")

        elif choice == "5":
            list_processes()

        elif choice == "6":
            show_notifications()

        elif choice == "7":
            print("Logging out...")
            notifications.append("User logged out at " + time.ctime())
            break

        else:
            print("Invalid choice. Please try again.")

    print("Exiting Python OS Simulation. Goodbye!")


# Run the program
if __name__ == "__main__":
    main()