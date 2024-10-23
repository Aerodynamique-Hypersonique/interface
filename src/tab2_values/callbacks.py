from src.layout import *

def define_callbacks2(app):
    @app.callback(
        Output('ok-button-value', 'n_clicks'),
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
        pass

    @app.callback(
        Output('input-pressure', 'value'),
        Output('input-temperature', 'value'),
        Output('input-density', 'value'),
        Output('input-viscosity', 'value'),
        Input('input-altitude', 'value'),
        prevent_initial_call=True
    )
    def update_values(_altitude):
        air = Physics.Air()
        data = air.get_atm_at_z(_altitude)

        return data['pressure'], data['temperature'], data['density'], data['viscosity']