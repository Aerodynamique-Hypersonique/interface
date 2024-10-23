from dash import no_update
import numpy as np
from src.layout import *


def define_callbacks1(app):
    @app.callback(
        Output('shape-graphs', 'figure'),
        Output('profile-store', 'data'),
        Input('ok-button-profile', 'n_clicks'),
        State('input-length', 'value'),
        State('input-angle', 'value'),
        State('dropdown-shape', 'value'),
        State('div-dynamic-components', 'children'),
        prevent_initial_call=True
    )
    def plot_shape(_n_clicks, _length, _angle, _shape, _dyn_attributes):
        if _shape == 'Conique':
            radius_nose = _dyn_attributes['props']['children'][1]['props']['value']  # Retrieve the radius_nose value
            if radius_nose is not None:
                profile = Profile.Conical(_angle, radius_nose, _length)

                figure = go.Figure(data=go.Scatter(x=np.concatenate((profile.get_x_nose(), profile.get_x())),
                                                   y=np.concatenate((profile.get_y_nose(), profile.get_y()))),
                                   layout=dark_graph_layout)
                figure.add_trace(go.Scatter(x=np.concatenate((profile.get_x_nose(), profile.get_x())),
                                            y=np.concatenate((-profile.get_y_nose(), -profile.get_y()))))
                return figure, profile.to_json()

        elif _shape == 'Parabolique':
            k = _dyn_attributes['props']['children'][1]['props']['value']  # Retrieve the k value
            if k is not None:
                profile = Profile.Parabolic(_angle, _length, k)

                figure = go.Figure(data=go.Scatter(x=profile.get_x(), y=profile.get_y()),
                                   layout=dark_graph_layout)
                figure.add_trace(go.Scatter(x=profile.get_x(), y=-profile.get_y()))
                print(profile.to_json())
                return figure, profile.to_json()

        return no_update

    @app.callback(
        Output('div-dynamic-components', 'children'),
        Input('dropdown-shape', 'value'),
        prevent_initial_call=True
    )
    def change_attributes_fields(value):
        if value == 'Conique':
            return html.Div(children=[
                html.B("Rayon de la tête", id='text-input-radius-front'),
                dcc.Input(id='input-radius-front', placeholder='Entrez le rayon de la tête', type='number')])
        elif value == 'Parabolique':
            return html.Div(children=[
                html.B("Type de la parabole"),
                dcc.Slider(0, 1, 0.01,
                           value=0,
                           marks={0: '0.0', 1: '1.0', **{i / 10: f'{i / 10:.1f}' for i in range(1, 10)}},
                           included=False,
                           id='input-parabolic-k')
            ])

        return html.Div()
