import base64
import io
import webbrowser

from dash import no_update
import numpy as np
from src.layout import *
import pandas as pd
import re


def define_callbacks1(app):
    @app.callback(
        Output('shape-graphs', 'figure', allow_duplicate=True),
        Output('profile-store', 'data'),
        Input('ok-button-profile', 'n_clicks'),
        State('input-length', 'value'),
        State('input-angle', 'value'),
        State('dropdown-shape', 'value'),
        State('div-dynamic-components', 'children'),
        prevent_initial_call=True,
        allow_duplicate=True
    )
    def plot_shape(_n_clicks, _length, _angle, _shape, _dyn_attributes):
        if _shape == 'Conique':
            radius_nose = _dyn_attributes['props']['children'][1]['props']['value']  # Retrieve the radius_nose value
            if radius_nose is not None:
                profile = Profile.Conical(_angle, radius_nose, _length)
                profile.x_nose = profile.x_nose - np.min(profile.x_nose)
                profile.x = profile.x - np.min(profile.x_nose)

                figure = go.Figure(data=go.Scatter(x=np.concatenate((profile.get_x_nose(), profile.get_x())),
                                                   y=np.concatenate((profile.get_y_nose(), profile.get_y())),
                                                   name='profil'),
                                   layout=dark_graph_layout)
                figure.add_trace(go.Scatter(x=np.concatenate((profile.get_x_nose(), profile.get_x())),
                                            y=np.concatenate((-profile.get_y_nose(), -profile.get_y())),
                                            name='profil (symétrie)'))

                return figure, profile.to_json()

        elif _shape == 'Parabolique':
            k = _dyn_attributes['props']['children'][1]['props']['value']  # Retrieve the k value
            if k is not None:
                profile = Profile.Parabolic(_angle, _length, k)

                figure = go.Figure(data=go.Scatter(x=profile.get_x(), y=profile.get_y()),
                                   layout=dark_graph_layout)
                figure.add_trace(go.Scatter(x=profile.get_x(), y=-profile.get_y()))
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

    @app.callback(
        Output('shape-graphs', 'figure', allow_duplicate=True),
        Output('error-text-upload', 'style'),
        Output('error-text-upload', 'children'),
        Input('upload-profile', 'contents'),
        State('upload-profile', 'filename'),
        prevent_initial_call=True
    )
    def update_figure(_content, _filename):
        # CSV file : X,Y\newline
        content_type, content_string = _content.split(',')
        decoded = base64.b64decode(content_string)
        if _filename.split('.')[1] == 'csv':
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep='\t')
            columns_name = df.columns.tolist()
            pattern_x = re.compile(r'.*?x.*?')
            pattern_y = re.compile(r'.*?y.*?')

            x_col = -1
            y_col = -1
            for(index, item) in enumerate(columns_name):
                item = item.lower()
                if pattern_x.match(item):
                    x_col = index
                elif pattern_y.match(item):
                    y_col = index

            if x_col > -1 and y_col > -1: # x and y or x and y and z (in case)
                x = df.iloc[:,x_col].to_numpy()
                y = df.iloc[:,y_col].to_numpy()

                figure = go.Figure(data=go.Scatter(x=x, y=y),
                                   layout=dark_graph_layout)
                return figure, {'visibility': 'hidden'}, ""
            else:
                return go.Figure(layout=dark_graph_layout), {}, "Erreur! Format tabulaire avec au moins deux colonnes contenant le titre 'x' et 'y'"



        return no_update

    @app.callback(
        Output('help-button-profile', 'n_clicks'),
        Input('help-button-profile', 'n_clicks'),
        prevent_initial_call=True
    )
    def get_help(_n_clicks):
        if _n_clicks > 0:
            webbrowser.open_new_tab('helps/help_profile.html')
        return _n_clicks
