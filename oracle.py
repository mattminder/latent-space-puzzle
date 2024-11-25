from hashlib import sha256

FIRST_NAME_HASH = '522ce7057fd0523adcd6672db24bb671d09d1ffa2f1e7c97c13e6c68ae6fcb13'
LAST_NAME_HASH = 'd8d6703ff595e5e938d82b59e69d6646e6118811078bb005a271a9184e5e4996'

def oracle(first_name, last_name):
    """Returns True if the right name was guessed."""

    def hash_(string):
        encoded = bytes(string.lower().strip(), "latin")
        return sha256(encoded).hexdigest()

    fn_correct = hash_(first_name) == FIRST_NAME_HASH
    ln_correct = hash_(last_name) == LAST_NAME_HASH

    return fn_correct and ln_correct
