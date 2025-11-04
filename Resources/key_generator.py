import os

# Generate a random secret key
secret_key = os.urandom(32)
print(secret_key.hex())
