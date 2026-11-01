"""Print the foot switch states. Press each foot to check left/right."""

import time

from _common import load_config

from odm_maix.feet_contacts import FeetContacts

feet = FeetContacts(load_config())
while True:
    left, right = feet.get()
    print(f"left {'DOWN' if left else 'up  '}   right {'DOWN' if right else 'up  '}", end="\r")
    time.sleep(0.05)
