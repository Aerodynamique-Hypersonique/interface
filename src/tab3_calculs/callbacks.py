from dash import no_update, ALL, ctx

from src.layout import *
import numpy as np
import json

from src.objects.Physics import ATM_LAYER_ALT
from src.objects.Profile import load_profile_from_dict


def define_callbacks3(app):
    @app.callback(
        Output('results-graphs', 'figure'),
        Input('ok-button-calcul', 'n_clicks'),
        State('profile-store', 'data'),
        State('physics-store', 'data'),
        prevent_initial_call=True
    )
    def calcul(_n_clicks, _profile_dict, _physics_dict):
        return no_update
        """
        profile: Profile.Profile = load_profile_from_dict(json.loads(_profile_dict))
        physics = Physics.Physics()
        physics.from_dict(json.loads(_physics_dict))

        x = profile.get_x()
        y = profile.get_y()

        rho = physics.get_rho()
        mu = physics.get_mu()

        local_reynolds = physics.get_local_reynolds(x)
        print(physics.atm.pressure)

        boundary_layer = ((x / np.sqrt(local_reynolds)) *
                          (np.sqrt((Physics.ATM_SEA_LEVEL['density'] * mu) / (
                                      rho * Physics.ATM_SEA_LEVEL['dynamic_visco']))))

        fig = go.Figure(data=go.Scatter(x=x, y=y, name='Profil'),
                        layout=dark_graph_layout)

        fig.add_trace(go.Scatter(x=x, y=boundary_layer + y, name='Couche Limite', line={'dash': 'dash'}))

        return fig
        """

    @app.callback(
        Output('highlight-store', 'data'),
        Input({'type': 'grid-item', 'index': ALL}, 'n_clicks'),
        prevent_initial_call=True
    )
    def highlight_graphs(_n_clicks):
        if ctx.triggered:
            clicked_index = ctx.triggered[0]['prop_id'].split('.')[0].split('index":')[1].split(',')[0]
            return clicked_index


    @app.callback(
        Output({'type': 'grid-item', 'index': ALL}, 'style'),
        Output({'type': 'graph-grid', 'index': ALL}, 'figure'),
        Input('highlight-store', 'data'),
        prevent_initial_call=True
    )
    def update_graphs(_hightlight_index):
        return no_update

