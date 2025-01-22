import hashlib
import os
import binascii

def hash_password(password: str) -> tuple[str, str]:
    """
    Hash a password with SHA-256 and a random salt.
    Returns (salt, hashed_password) as hex strings.
    """
    # Generate a random salt
    salt = os.urandom(32)
    # Convert the salt to hex for storage
    salt_hex = binascii.hexlify(salt).decode('utf-8')
    
    # Hash the password with the salt
    pwdhash = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt, 
        100000,  # Number of iterations
        dklen=128  # Length of the derived key
    )
    
    # Convert the hash to hex for storage
    hash_hex = binascii.hexlify(pwdhash).decode('utf-8')
    
    return salt_hex, hash_hex

def verify_password(stored_salt: str, stored_password_hash: str, provided_password: str) -> bool:
    """
    Verify a stored password against a provided password
    """
    # Convert stored salt from hex back to bytes
    salt = binascii.unhexlify(stored_salt.encode('utf-8'))
    
    # Hash the provided password with the stored salt
    pwdhash = hashlib.pbkdf2_hmac(
        'sha256',
        provided_password.encode('utf-8'),
        salt,
        100000,  # Same number of iterations as in hash_password
        dklen=128  # Same length as in hash_password
    )
    
    # Convert to hex for comparison
    provided_hash_hex = binascii.hexlify(pwdhash).decode('utf-8')
    
    # Compare the hashes
    return provided_hash_hex == stored_password_hash