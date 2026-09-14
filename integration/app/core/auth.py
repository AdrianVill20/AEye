def authenticate(user_id: str, password: str, role: str) -> bool:
    """PLACEHOLDER authentication.

    The real version will verify the school-issued ID and password
    against the central server (Module 3). For now it only checks that
    both fields are filled, so the login flow can be built and shown.

    NOT secure — this does not validate real accounts yet.
    """
    return bool(user_id.strip()) and bool(password.strip())