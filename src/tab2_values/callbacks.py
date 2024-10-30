from src.layout import *
from dash import no_update

def define_callbacks2(app):
    @app.callback(
        Output('physics-store', 'data'),
        Input('ok-button-value', 'n_clicks'),
        State('input-altitude', 'value'),
        State('input-mass-mol', 'value'),
        State('input-pressure', 'value'),
        State('input-temperature', 'value'),
        State('input-velocity-x', 'value'),
        State('input-velocity-y', 'value'),
        State('input-density', 'value'),
        State('input-viscosity', 'value'),
        State('input-gamma', 'value'),
        State('input-gravity', 'value'),
        prevent_initial_call=True
    )
    def set_values(_n_clicks, _altitude, _m_mol, _pressure, _temperature, _velo_x, _velo_y, _density, _viscosity,
                      _gamma, _g):
        physics = Physics.Physics()

        physics.atm.altitude    = _altitude
        physics.atm.m_mol       = _m_mol
        physics.atm.pressure    = _pressure
        physics.atm.temperature = _temperature
        physics.atm.density     = _density
        physics.atm.viscosity   = _viscosity
        physics.atm.gamma       = _gamma
        physics.gravity         = _g

        physics.velocity_x = _velo_x
        physics.velocity_y = _velo_y

        return physics.to_json()

    @app.callback(
        Output('input-pressure', 'value'),
        Output('input-temperature', 'value'),
        Output('input-density', 'value'),
        Output('input-viscosity', 'value'),
        Input('input-altitude', 'value'),
        State('input-gravity', 'value'),
        State('modify-values-check', 'value'),
        prevent_initial_call=True
    )
    def update_values(_altitude, _gravity, _modify):
        if _altitude is None or _modify is not None and 'modify' in _modify:
            return no_update

        atm = Physics.Atmosphere()
        data = atm.get_atm_at_z(_altitude, _gravity)
        return data['pressure'], data['temperature'], data['density'], data['viscosity']

    @app.callback(
        Output('input-mass-mol', 'disabled'),
        Output('input-pressure', 'disabled'),
        Output('input-temperature', 'disabled'),
        Output('input-density', 'disabled'),
        Output('input-viscosity', 'disabled'),
        Output('input-gamma', 'disabled'),
        Output('input-gravity', 'disabled'),
        Input('modify-values-check', 'value'),
        prevent_initial_call=True
    )
    def modify_specific_values(_check_value):
        if 'modify' in _check_value:
            return [False] * 7
        return [True] * 7
