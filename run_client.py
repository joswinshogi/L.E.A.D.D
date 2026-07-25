import sys

from src.flower_client import start_client

if len(sys.argv) != 2:
    print("Usage: python run_client.py client1")
    sys.exit()

client_name = sys.argv[1]

client_path = f"clients/{client_name}"

start_client(client_path)