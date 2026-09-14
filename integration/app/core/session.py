from dataclasses import dataclass


@dataclass
class Session:
    """Who is currently signed in. Binds the whole exam session to a
    school ID so evidence (screenshots/incidents) can be tagged with it
    later (WBS 2.0: 'bind each session to a student ID for evidence
    tagging'). Module 3 reads this when logging incidents.
    """
    user_id: str
    role: str   # "student" or "proctor"