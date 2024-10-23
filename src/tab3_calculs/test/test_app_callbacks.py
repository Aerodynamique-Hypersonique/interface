from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

from src.tab3_calculs.callbacks import calcul_test_unit

def test_calcul_callback():
    with open('profile_test_json', 'r') as file:
        profile = file.read().rstrip()
    with open('physics_test_json', 'r') as file:
        physics = file.read().rstrip()

    output = calcul_test_unit(profile, physics)
    print(output)

